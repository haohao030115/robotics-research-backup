"""Isolated alignment retraining; original V2 causal dataset semantics stay intact."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.tokens import TokenDataset
from utils.v2_config import load_v2_config
import trainers.token_training as training


class DiagnosticOffsetDataset(TokenDataset):
    def __init__(self,episodes,image_size,history_length,k,offset):
        super().__init__(episodes,image_size,history_length,k,max(1,offset))
        if offset<0:raise ValueError('Nonnegative diagnostic offset required')
        self.target_offset=offset
        self.index=[(i,j,int(t)) for i,e in enumerate(episodes) for j,t in enumerate(e.anchors) if t+offset+k<=len(e.times)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','oracle-report','overfit-report','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--offsets',type=int,nargs='+',default=[0,2,3]);args=p.parse_args()
    config=load_v2_config(args.config)
    def dataset(episodes,c):
        m=c['model'];return DiagnosticOffsetDataset(episodes,m['image_size'],m['history_length'],m['token_chunk_size'],c['data']['target_offset'])
    training.dataset_for=dataset
    for offset in args.offsets:
        config['data']['target_offset']=offset;config['training']['output_dir']=str(Path(args.output)/f'offset_{offset}')
        report=training.train_tokens(config,args.oracle_report,'heldout',args.overfit_report)
        print('offset',offset,'val',report['val']['token_mae'],report['val']['hand_mae'],flush=True)


if __name__=='__main__':main()
