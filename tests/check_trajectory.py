"""Compare a completed example with the included frozen trajectory."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=['gradient','strong_lvz','crustal','field'])
    parser.add_argument('--run',required=True)
    parser.add_argument('--tolerance',type=float,default=1e-10)
    args=parser.parse_args()
    reference=Path(__file__).resolve().parents[1]/'examples/results'/args.case
    actual=Path(args.run)
    differences={}
    for method in ['raw','sn']:
        with np.load(reference/(method+'_trajectory.npz')) as old,np.load(actual/method/'trajectory.npz') as new:
            np.testing.assert_array_equal(old['updates'],new['updates'])
            differences[method]={}
            for key in ['vs','h','training_loss']:
                assert old[key].shape==new[key].shape,(method,key)
                delta=float(np.max(abs(old[key]-new[key])))
                differences[method][key]=delta
                assert delta<=args.tolerance,(method,key,delta)
    old=json.loads((reference/'switch_diagnostics.json').read_text())
    new=json.loads((actual/'sn/switch_diagnostics.json').read_text())
    assert old['switch_updates']==new['switch_updates']
    print(json.dumps(dict(passed=True,maximum_absolute_differences=differences),indent=2))


if __name__=='__main__':main()
