import argparse
from .hand import calibrate

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=calibrate(a.dataset,c['manifest'],a.output)
    print({k:v for k,v in cfg.items() if k not in ('expert_sources','trajectory_reference_ids')},flush=True)
if __name__=='__main__':main()
