import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v3.training import load_checkpoint
from v3.rollout import run
from datasets.tokens import load_token_episodes
from utils.runtime import initialize, cleanup


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--episode-id',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','musa'],default='musa')
    p.add_argument('--vision-worker',choices=['process','thread'],default='process')
    p.add_argument('--render-height',type=int,default=120);p.add_argument('--render-width',type=int,default=160)
    p.add_argument('--diagnostic-continue-on-miss',action='store_true',help='NONCOMPLIANT diagnostic video: continue and report stale slots; default aborts')
    args=p.parse_args();rt=initialize(args.device,38)
    try:
        _,saved=load_checkpoint(args.checkpoint,rt.device)
        c=copy.deepcopy(saved['config']);c['training']['device']=args.device;c['rollout']['strict_timing']=not args.diagnostic_continue_on_miss
        c['rollout'].update(vision_worker=args.vision_worker,render_height=args.render_height,render_width=args.render_width)
        data=copy.deepcopy(c['data']);data['episode_ids']=[args.episode_id]
        report=run(c,args.checkpoint,load_token_episodes(data)[0],args.output)
        if report['failure'] or not report['timing_contract_pass']:raise SystemExit(2)
    finally:cleanup(rt)

if __name__=='__main__':main()
