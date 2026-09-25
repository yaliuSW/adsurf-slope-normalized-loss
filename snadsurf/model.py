"""Rayleigh-wave equation and its local phase-velocity slope.

Arrays use km, km/s and g/cm³. Models have shape (starts, layers), observed
velocities and periods have shape (starts, points). The last layer is a halfspace.
"""
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
from ADsurf._cps import _surf96_vectorAll_gpu as vector
from ADsurf._cps import _surf96_matrixAll_gpu as matrix


def brocher_vp_rho(vs):
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    return vp, rho


class DispersionModel(torch.nn.Module):
    def __init__(self, h, vp, vs, rho, velocity_grid, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.b = torch.nn.Parameter(vs)
        self.d = torch.nn.Parameter(h) if config['vary_h'] else h
        self.a, self.rho, self.clist = vp, rho, velocity_grid
        self.llw = 0 if vs[0, 0] <= 0 else -1
        n, layers = vs.shape
        laplacian = (np.diag(-np.ones(layers-1), 1)
                     + np.diag(-np.ones(layers-1), -1)
                     + np.diag(2*np.ones(layers)))
        laplacian[0, 0] = laplacian[-1, -1] = 1
        self.Lv = torch.as_tensor(np.repeat(laplacian[None].astype(np.float32), n, axis=0),
                                  dtype=vs.dtype, device=device)

    def elastic(self, vs=None):
        vs = self.b if vs is None else vs
        if self.config['elastic'] == 'Brocher':
            return brocher_vp_rho(vs)
        if self.config['elastic'] == 'Constant':
            return vs*self.config['vp_vs_ratio'], self.rho
        return self.a, self.rho

    def transform(self, residual):
        if self.config['mapped']:
            return torch.sign(residual) * (1.0 - (1e-1)**torch.abs(residual))
        return residual

    def equation(self, c, periods):
        vp, rho = self.elastic()
        if self.config['mapped']:
            # Repeated observation periods share exactly the same detached range.
            unique, inverse = torch.unique(periods[0], sorted=True, return_inverse=True)
            with torch.no_grad():
                det = matrix.dltar_matrix(self.clist, unique[None].expand(len(periods), -1),
                    self.d, vp, self.b, rho, 2, self.llw, device=self.device)
                scale = (det.max(1).values-det.min(1).values).clamp(min=1e-12)[:, inverse]
            return vector.dltar_vector(c, periods, self.d, vp, self.b, rho,
                                      2, self.llw, device=self.device)/scale
        return vector.dltar_vector(c, periods, self.d, vp, self.b, rho,
                                  2, self.llw, device=self.device)

    def slope(self, c, periods):
        # Keep the full period grid here to preserve the frozen implementation's
        # floating-point evaluation order. No model gradient passes through it.
        with torch.no_grad():
            vs, h = self.b.detach(), self.d.detach()
            vp, rho = self.elastic(vs)
            dc = torch.maximum(torch.full_like(c, 1e-4), c.abs()*1e-4)
            plus, minus = c+dc, (c-dc).clamp(min=1e-6)
            rp = vector.dltar_vector(plus, periods, h, vp, vs, rho, 2, self.llw, device=self.device)
            rm = vector.dltar_vector(minus, periods, h, vp, vs, rho, 2, self.llw, device=self.device)
            if self.config['mapped']:
                det = matrix.dltar_matrix(self.clist, periods, h, vp, vs, rho,
                                         2, self.llw, device=self.device)
                scale = (det.max(1).values-det.min(1).values).clamp(min=1e-12)
                rp, rm = rp/scale, rm/scale
            derivative = (self.transform(rp)-self.transform(rm))/(plus-minus).clamp(min=1e-12)
            if not torch.isfinite(derivative).all():
                raise FloatingPointError('Nonfinite phase-velocity derivative')
            return derivative.abs()

    def weights(self, c, periods, normalize_slope=True):
        slope = self.slope(c, periods)
        floor = (.05*torch.quantile(slope, .05, dim=1, keepdim=True)).clamp(min=1e-6)
        # Exponent zero is the declared control retaining only observed-velocity
        # scaling; the final SN-ADsurf algorithm always uses exponent one.
        return (1/(c.abs()*torch.maximum(slope, floor).pow(int(normalize_slope)))).clamp(max=30).detach()

    def regularization(self):
        return self.config['damping']*(torch.matmul(self.Lv, self.b.unsqueeze(-1)).squeeze(-1)
                                      / self.b.shape[1]).abs().sum(1)
