"""Small offline adapter/hand smoke CLI; native execution remains producer-owned."""
from __future__ import annotations
import argparse, json
import numpy as np
from .adapter import UnifiedMotionAdapter
from .hand import HandMotionGenerator

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--producer',choices=['kimodo','ardy'],required=True)
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--cube-xy',nargs=2,type=float,default=[0.,0.])
    a=p.parse_args(argv); x=np.load(a.input); q=x['body_ref_q'] if 'body_ref_q' in x else x['q']
    result=UnifiedMotionAdapter().adapt(q,task={'cube_xy':a.cube_xy},seed=a.seed,producer=a.producer)
    UnifiedMotionAdapter.save(result,a.output); np.save(a.output+'/hand_ref_q.npy',HandMotionGenerator().generate(np.arange(len(q))/50.,seed=a.seed)[0])
    print(json.dumps(result.corrections,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
