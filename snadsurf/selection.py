"""Whole-branch dispersion screening, separate from model updates.

Synthetic examples prescribe modal identities. The field example prescribes
the fundamental branch and compares ordered assignments for its two higher
groups. Reference Vs profiles are never used here.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import json
import multiprocessing
import numpy as np
from .model import brocher_vp_rho


def spectrum(vs, h, frequencies, config, vp_fixed, rho_fixed, limit):
    import qedispinv_forward as forward
    if config['elastic']=='Brocher': vp, rho = brocher_vp_rho(vs)
    elif config['elastic']=='Constant': vp, rho = vs*config['vp_vs_ratio'], rho_fixed
    else: vp, rho = vp_fixed, rho_fixed
    model = np.asfortranarray(np.column_stack([
        np.arange(1,len(vs)+1), np.r_[0,np.cumsum(h[:-1])], rho, vs, vp]))
    curves = np.atleast_2d(np.asarray(forward.dispersion(
        model, np.asfortranarray(frequencies), limit, 'rayleigh')))
    result = np.full((len(frequencies), limit+1), np.nan)
    for f,c,mode in curves:
        if np.isfinite([f,c,mode]).all() and c>0:
            j = int(np.argmin(abs(frequencies-f)))
            if abs(frequencies[j]-f)>=1e-6: raise ValueError('Forward frequency mismatch')
            result[j,int(round(mode))] = c
    return result


def score_spectrum(sp, obs, index, field=False):
    groups = obs[:,2].astype(int)
    if field:
        if set(groups)!={0,1,2}: raise ValueError('Field assignment expects groups 0, 1 and 2')
        modes = np.full(len(obs),-1,int); modes[groups==0] = 0
        pred = np.full(len(obs),np.nan); pred[groups==0] = sp[index[groups==0],0]
        errors = {}
        for g in [1,2]:
            for mode in range(1,sp.shape[1]):
                c = sp[index[groups==g],mode]
                if np.isfinite(c).all(): errors[g,mode] = float(np.mean((c-obs[groups==g,1])**2))
        pairs = [(errors[1,a]+errors[2,b],a,b) for a in range(1,sp.shape[1])
                 for b in range(a+1,sp.shape[1]) if (1,a) in errors and (2,b) in errors]
        if pairs:
            _,a,b = min(pairs)
            for g,mode in [(1,a),(2,b)]:
                mask = groups==g; modes[mask] = mode; pred[mask] = sp[index[mask],mode]
    else:
        modes = groups.copy(); pred = sp[index,modes]
    row = dict(status='missing_root',coverage=int(np.isfinite(pred).sum()))
    if np.isfinite(pred).all():
        margins = []
        for g in ([0] if field else np.unique(groups)):
            mask = groups==g
            residual = (sp[index[mask]]-obs[mask,1,None])*1000
            errors = np.sqrt(np.mean(residual**2,axis=0))
            errors = np.where(np.isfinite(errors),errors,np.inf)
            assigned = float(errors[g]); errors[g] = np.inf
            margins.append(float(errors.min()-assigned))
        margin = min(margins)
        row.update(whole_branch_margin_m_s=margin,
            status='wrong_branch' if margin<-.001 else ('ambiguous_branch' if margin<=.001 else 'accepted'))
        errors = []
        for g in np.unique(groups):
            error = float(np.sqrt(np.mean((pred[groups==g]-obs[groups==g,1])**2))*1000)
            row[f'group{g}_RMSE_m_s'] = error; errors.append(error)
        weights = np.array([4 if g==0 else 1 for g in np.unique(groups)])
        row['weighted_RMSE_m_s'] = float(np.sqrt(np.sum(weights*np.array(errors)**2)/weights.sum()))
        row['all_RMSE_m_s'] = float(np.sqrt(np.mean((pred-obs[:,1])**2))*1000)
    return row,pred,modes


def initialize_worker(obs, config, vp, rho):
    global OBS,CONFIG,VP,RHO,FREQUENCIES,INDEX
    OBS,CONFIG,VP,RHO = obs,config,vp,rho
    FREQUENCIES,INDEX = np.unique(obs[:,0],return_inverse=True)


def evaluate_model(model):
    try:
        vs,h = model
        field = CONFIG['case']=='field'
        limit = 40 if field else int(OBS[:,2].max())+1
        sp = spectrum(vs,h,FREQUENCIES,CONFIG,VP,RHO,limit)
        return score_spectrum(sp,OBS,INDEX,field)
    except Exception as exc:
        return dict(status='forward_exception',error=str(exc)),np.full(len(OBS),np.nan),np.full(len(OBS),-1)


def select(arrays, config, run_directory, workers=6):
    """Screen every saved model, then select raw, SN-only and pooled candidates."""
    root = Path(run_directory); out = root/'selection'; out.mkdir(exist_ok=False)
    obs = arrays['obs']; models,metadata = [],[]
    for method in ['raw','sn']:
        path = root/method
        if not json.loads((path/'COMPLETE.json').read_text())['completed']:
            raise ValueError('Incomplete optimization path')
        with np.load(path/'trajectory.npz') as trajectory:
            for k,update in enumerate(trajectory['updates']):
                for sid,(vs,h) in enumerate(zip(trajectory['vs'][k],trajectory['h'][k])):
                    if config['case']=='gradient' and sid>0: continue
                    metadata.append(dict(candidate=len(metadata),method=method,start=sid,update=int(update)))
                    models.append((vs.copy(),h.copy()))
    vp = arrays['starts_vp'][0] if 'starts_vp' in arrays else None
    rho = arrays['starts_rho'][0] if 'starts_rho' in arrays else None
    init = (obs,config,vp,rho)
    if workers==1:
        initialize_worker(*init); outputs = list(map(evaluate_model,models))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker,initargs=init) as pool:
            outputs = list(pool.map(evaluate_model,models,chunksize=12))
    rows = [dict(**meta,**value[0]) for meta,value in zip(metadata,outputs)]
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'candidate_scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if any(r['status']=='forward_exception' for r in rows):
        raise RuntimeError('Forward evaluation failed; see candidate_scores.csv')
    summary = {}
    for label,methods in [('raw',{'raw'}),('sn_only',{'sn'}),('retained',{'raw','sn'})]:
        valid = [r for r in rows if r['method'] in methods and r['status']=='accepted']
        if not valid:
            summary[label] = {'status':'no_eligible_model'}; continue
        winner = min(valid,key=lambda row:row['weighted_RMSE_m_s']);i=winner['candidate']
        best = {}
        for row in valid:
            sid=row['start']
            if sid not in best or row['weighted_RMSE_m_s']<best[sid]['weighted_RMSE_m_s']:best[sid]=row
        summary[label] = dict(winner,eligible_starts=len(best))
        np.savez_compressed(out/(label+'_selected.npz'),vs=models[i][0],h=models[i][1],
                            obs=obs,pred=outputs[i][1],modes=outputs[i][2])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if summary['retained'].get('status')=='accepted' and summary['raw'].get('status')=='accepted':
        assert summary['retained']['weighted_RMSE_m_s']<=summary['raw']['weighted_RMSE_m_s']
    return summary
