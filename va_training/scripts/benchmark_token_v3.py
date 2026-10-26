"""Forward/preprocessing latency on a real current observation; no fake dispatch."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from v3.training import load_checkpoint
from v3.inference import CurrentPolicy,latency_report
from v3.data import FIELDS
from datasets.tokens import load_token_episodes
from utils.runtime import initialize,cleanup
from utils.config import dump_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','musa'],default='cpu');p.add_argument('--iterations',type=int,default=100);args=p.parse_args()
    if args.iterations<1:raise ValueError('Positive iterations required')
    rt=initialize(args.device,38)
    try:
        model,saved=load_checkpoint(args.checkpoint,rt.device);policy=CurrentPolicy(model)
        c=saved['config']['data'].copy();c['episode_ids']=[saved['split']['val'][0]];e=load_token_episodes(c)[0]
        with np.load(e.image_path) as z:rgb=z['head_rgb'][0].copy()
        state={k:e.state[k][e.anchors[0]] for k in FIELDS}
        rows=[]
        for i in range(args.iterations+10):
            _,timing=policy.predict(rgb,state)
            if i>=10:rows.append(timing)
        report=dict(device=str(rt.device),iterations=args.iterations,warmup=10,units='milliseconds',stages=latency_report(rows),
            scope='Real saved observation preprocessing + VA. Live rendering, dispatch and observation-to-command measured by closed-loop runner.',
            parameters=sum(p.numel() for p in model.parameters()))
        dump_json(args.output,report);print(report,flush=True)
    finally:cleanup(rt)

if __name__=='__main__':main()
