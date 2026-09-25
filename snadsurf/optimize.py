"""ADsurf and one-way slope-normalized refinement with shared settings."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from .model import DispersionModel, brocher_vp_rho, vector, matrix
from .switching import PlateauSwitch


def optimize(arrays, config, output, method='sn', device='cpu', iterations=400,
             normalize_slope=True):
    """Save a paired-comparison path; no reference model or root search is used.

    Required arrays: obs (Hz, km/s, group), starts_vs, starts_h, starts_vp,
    starts_rho, lower_vs, upper_vs and clist. Thickness inversion also needs
    lower_h and upper_h. Group zero has weight four; other groups have weight one.
    """
    if method not in ('raw', 'sn') or iterations < 2:
        raise ValueError('Expected method raw/sn and at least two updates')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    cfg = dict(config)
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    def preserve(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float64)
    vector.numpy2tensor = preserve; matrix.numpy2tensor = preserve
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float64, device=device)
    a = arrays
    vs, h, obs = a['starts_vs'].copy(), a['starts_h'].copy(), a['obs']
    n = len(vs)
    c = tensor(np.tile(obs[:, 1], (n, 1)))
    periods = tensor(np.tile(1/obs[:, 0], (n, 1)))
    vp, rho = brocher_vp_rho(vs) if cfg['elastic']=='Brocher' else (a['starts_vp'], a['starts_rho'])
    model = DispersionModel(tensor(h), tensor(vp), tensor(vs.copy()), tensor(rho),
                            tensor(np.tile(a['clist'], (n, 1))), cfg, device)
    groups = np.unique(obs[:, 2]).astype(int)
    masks = [torch.as_tensor(obs[:, 2]==g, device=device) for g in groups]
    group_weights = tensor([4. if g==0 else 1. for g in groups])
    point_weights = torch.zeros_like(c)
    for mask, weight in zip(masks, group_weights):
        point_weights[:, mask] = weight/group_weights.sum()/mask.sum()
    def reduce(values):
        return (values*point_weights).sum(1)
    def components(weights):
        residual = model.transform(model.equation(c, periods)).abs()
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Nonfinite dispersion-equation residual')
        return reduce(residual), reduce((residual*weights).clamp(max=100))
    lower, upper = tensor(a['lower_vs']), tensor(a['upper_vs'])
    def project():
        with torch.no_grad():
            if not torch.isfinite(model.b).all(): raise FloatingPointError('Nonfinite Vs')
            model.b.copy_(torch.maximum(lower, torch.minimum(upper, model.b)))
            model.b[:, -1].copy_(torch.maximum(model.b[:, -1], model.b[:, -2]))
            if cfg['vary_h']:
                if not torch.isfinite(model.d).all(): raise FloatingPointError('Nonfinite thickness')
                model.d.copy_(torch.maximum(tensor(a['lower_h']), torch.minimum(tensor(a['upper_h']), model.d)))
                model.d[:, -1] = 0.
                if cfg.get('nondecreasing_thickness', False):
                    values = model.d[:, :-1].detach().cpu().numpy().copy()
                    for row in values:
                        levels, counts = [], []
                        for value in row:
                            levels.append(float(value)); counts.append(1)
                            while len(levels)>1 and levels[-2]>levels[-1]:
                                count = counts[-2]+counts[-1]
                                value = (levels[-2]*counts[-2]+levels[-1]*counts[-1])/count
                                levels[-2:] = [value]; counts[-2:] = [count]
                        row[:] = np.repeat(levels, counts)
                    if not (np.all(values>=a['lower_h'][:, :-1]-1e-12)
                            and np.all(values<=a['upper_h'][:, :-1]+1e-12)):
                        raise ValueError('Monotonic projection conflicts with thickness bounds')
                    model.d[:, :-1].copy_(tensor(values))
    project()
    with torch.no_grad():
        initial_weights = model.weights(c, periods, normalize_slope)
        raw0, sn0 = components(initial_weights)
        scale = float(raw0[0]/sn0[0])
        if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid objective calibration')
    controller = PlateauSwitch(n, **cfg.get('plateau_options', {})) if method=='sn' else None
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr_initial'], betas=(.9, .999))
    cfg.update(method=method, iterations=iterations, scale=scale,
               slope_refresh=1, slope_gradient=False, normalize_slope=normalize_slope,
               group_weights={str(g):float(w) for g,w in zip(groups,group_weights.cpu())})
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    steps, velocities, thicknesses, records, training = [], [], [], [], []
    started = time.perf_counter()
    def save():
        np.savez_compressed(out/'trajectory.npz', updates=steps, vs=velocities, h=thicknesses, training_loss=training)
        (out/'checkpoints.json').write_text(json.dumps(records))
        if controller: (out/'switch_diagnostics.json').write_text(json.dumps(controller.report(), indent=2))
    for k in range(iterations+1):
        progress = .5*(1-np.cos(np.pi*min(k,iterations-1)/(iterations-1)))
        lr = cfg['lr_initial']*(1-.9*progress)
        newly = np.zeros(n, dtype=bool)
        fraction = 0.
        if controller:
            with torch.no_grad():
                monitor = reduce(model.transform(model.equation(c, periods)).abs()).cpu().numpy()
            newly = controller.observe(k, monitor, groups.tolist(), allow_switch=k<iterations)
            fraction = tensor(controller.switched.astype(float))
        if k%20==0 or k==iterations or newly.any():
            with torch.no_grad():
                fresh = model.weights(c, periods, normalize_slope)
                raw, sn = components(fresh); regularization = model.regularization()
                records.append(dict(update=k, weight=fraction.cpu().tolist() if controller else 0., lr=lr,
                    raw=raw.cpu().tolist(), sn_scaled=(scale*sn).cpu().tolist(),
                    regularization=regularization.cpu().tolist(),
                    mixed=((1-fraction)*raw+fraction*scale*sn+regularization).cpu().tolist()))
                steps.append(k); velocities.append(model.b.detach().cpu().numpy().copy())
                thicknesses.append(model.d.detach().cpu().numpy().copy())
            save()
            if k%100==0 or k==iterations:
                print(json.dumps(dict(method=method, update=k, elapsed_s=time.perf_counter()-started)), flush=True)
        if k==iterations: break
        weights = model.weights(c, periods, normalize_slope)
        assert not weights.requires_grad and weights.grad_fn is None
        raw, sn = components(weights)
        loss = (1-fraction)*raw+fraction*scale*sn+model.regularization()
        optimizer.param_groups[0]['lr'] = lr; optimizer.zero_grad(); loss.sum().backward()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite model gradient')
                parameter.grad.mul_((10/parameter.grad.norm(dim=1,keepdim=True).clamp(min=1e-30)).clamp(max=1))
        optimizer.step(); project(); training.append(loss.detach().cpu().numpy())
    save()
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=True, updates=iterations,
        starts=n, seconds=time.perf_counter()-started, detached_weight_refreshes=iterations), indent=2))
