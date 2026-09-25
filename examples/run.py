"""Run a frozen example using the final ADsurf/SN-ADsurf pair."""
from pathlib import Path
import argparse
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from snadsurf.optimize import optimize
from snadsurf.selection import select


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='gradient')
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--iterations',type=int,default=400)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--skip-selection',action='store_true',help='Optimize only; no forward solver required')
    args=parser.parse_args()
    data=Path(__file__).parent/'data'
    with np.load(data/(args.case+'.npz')) as f:arrays=dict(f)
    config=json.loads((data/(args.case+'.json')).read_text())
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    for method in ['raw','sn']:
        optimize(arrays,config,out/method,method,args.device,args.iterations)
    with np.load(out/'raw/trajectory.npz') as raw,np.load(out/'sn/trajectory.npz') as sn:
        switch=json.loads((out/'sn/switch_diagnostics.json').read_text())['switch_updates']
        index={int(step):i for i,step in enumerate(raw['updates'])}
        for sid,transition in enumerate(switch):
            for j,step in enumerate(sn['updates']):
                if step not in index or (transition>=0 and step>transition):continue
                for key in ['vs','h']:
                    assert np.max(abs(raw[key][index[int(step)],sid]-sn[key][j,sid]))<1e-10
    if not args.skip_selection:
        print(json.dumps(select(arrays,config,out,args.workers),indent=2))
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,case=args.case,
        iterations=args.iterations,selection_completed=not args.skip_selection)))


if __name__=='__main__':main()
