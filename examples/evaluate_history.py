"""Evaluate physical dispersion errors along two saved paths from one start.

This diagnostic never changes optimization or final candidate eligibility.
Missing predictions are omitted within each group and their coverage is saved.
Every group must retain at least one point for a diagnostic score to be defined.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from snadsurf.selection import initialize_worker, evaluate_model

HERE = Path(__file__).resolve().parent


def available_point_score(obs, pred):
    """Keep the original group weights; average over available points per group."""
    errors, weights, coverage = [], [], {}
    for group in np.unique(obs[:, 2]).astype(int):
        mask = (obs[:, 2] == group) & np.isfinite(pred)
        coverage[f'group{group}_points'] = int(mask.sum())
        if mask.any():
            errors.append(float(np.mean(((pred[mask] - obs[mask, 1])*1000)**2)))
        else:
            errors.append(np.nan)
        weights.append(4 if group == 0 else 1)
    return float(np.sqrt(np.average(errors, weights=weights))), coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='strong_lvz')
    parser.add_argument('--run', help='New paired inversion directory')
    parser.add_argument('--start', type=int, help='Zero-based index; default: retained winner start')
    parser.add_argument('--out', required=True, help='Directory for history.csv and history.json')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    source = Path(args.run)/'selection' if args.run else HERE/'results'/args.case
    summary = json.loads((source/'summary.json').read_text())
    if isinstance(summary, list):
        summary = {row['method']: row for row in summary}
    start = args.start if args.start is not None else int(summary['retained']['start'])
    data_path = HERE/'data'/(args.case+'.npz')
    config_path = HERE/'data'/(args.case+'.json')
    arrays = np.load(data_path)
    config = json.loads(config_path.read_text())
    if not 0 <= start < len(arrays['starts_vs']):
        raise ValueError('Start index outside saved ensemble')
    switch_path = (Path(args.run)/'sn/switch_diagnostics.json' if args.run
                   else source/'switch_diagnostics.json')
    switch = json.loads(switch_path.read_text())['switch_updates'][start]
    models, metadata = [], []
    inputs = [data_path, config_path, source/'summary.json', switch_path]
    for method in ['raw', 'sn']:
        path = (Path(args.run)/method/'trajectory.npz' if args.run
                else source/(method+'_trajectory.npz'))
        inputs.append(path)
        with np.load(path) as tr:
            for k, step in enumerate(tr['updates']):
                metadata.append(dict(method=method, start=start, update=int(step)))
                models.append((tr['vs'][k, start].copy(), tr['h'][k, start].copy()))
    obs = arrays['obs']
    init = (obs, config, arrays['starts_vp'][0] if 'starts_vp' in arrays else None,
            arrays['starts_rho'][0] if 'starts_rho' in arrays else None)
    if args.workers == 1:
        initialize_worker(*init)
        outputs = list(map(evaluate_model, models))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_worker, initargs=init) as pool:
            outputs = list(pool.map(evaluate_model, models))
    rows = []
    for meta, (score, pred, _) in zip(metadata, outputs):
        if score['status'] == 'forward_exception':
            raise RuntimeError(score['error'])
        rmse, counts = available_point_score(obs, pred)
        rows.append(dict(**meta, weighted_RMSE_m_s=rmse,
            coverage=int(np.isfinite(pred).sum()), total_points=len(obs),
            selection_status=score['status'], **counts))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out/'history.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    info = dict(case=args.case, start=start, start_indexing='zero-based',
        start_choice='retained winner' if args.start is None else 'user supplied',
        switch_update=int(switch), raw_ensemble_best_RMSE_m_s=summary['raw']['weighted_RMSE_m_s'],
        retained_selected_update=summary['retained']['update'],
        missing_root_rule='Omit missing points within each group; require at least one point in every group. Diagnostic only; final selection still requires complete predictions.',
        comparison='Current checkpoint errors, not cumulative minima; both paths use the same start.',
        input_sha256={key: hashlib.sha256(p.read_bytes()).hexdigest() for key,p in zip(
            ['observations_and_starts','config','selection_summary','switch_diagnostics',
             'raw_trajectory','sn_trajectory'],inputs)})
    (out/'history.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(dict(case=args.case, start=start, switch_update=switch,
        checkpoints=len(rows), incomplete=sum(r['coverage'] < len(obs) for r in rows))))


if __name__ == '__main__':
    main()
