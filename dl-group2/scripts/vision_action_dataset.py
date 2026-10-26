"""NumPy map-style Dataset; compatible with torch DataLoader without importing torch.

One sample per saved RGB frame. RGB stays uint8 HWC; robot/action values are
float32 radians. Cube ground truth and overview camera are never model inputs.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np


def episode_split(episodes, validation_fraction=.2, seed=0):
    if not 0<=validation_fraction<1:raise ValueError('validation_fraction must be in [0,1)')
    ids=sorted(e['episode_id'] for e in episodes)
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate episode IDs')
    shuffled=np.random.default_rng(seed).permutation(ids).tolist()
    count=min(len(ids)-1,max(1,round(len(ids)*validation_fraction))) if len(ids)>1 and validation_fraction else 0
    return {'train':sorted(shuffled[count:]),'validation':sorted(shuffled[:count]),'seed':seed,
            'validation_fraction':validation_fraction,'unit':'whole_episode'}


class VisionActionDataset:
    def __init__(self,manifest,horizon=50,split='all',split_seed=0,validation_fraction=.2,
                 tail='pad',cache_episodes=2,split_file=None):
        self.manifest=Path(manifest).resolve();self.base=self.manifest.parent
        info=json.loads(self.manifest.read_text())
        if info['schema_version']!=3:raise ValueError('Requires schema v3')
        if not isinstance(horizon,(int,np.integer)) or horizon<1 or tail not in ('pad','drop') or cache_episodes<1:raise ValueError('Invalid horizon/tail/cache')
        if split not in ('all','train','validation'):raise ValueError('Invalid split')
        episodes=info['episodes']
        paths=[(self.base/e['episode_path']).resolve() for e in episodes]
        if len(paths)!=len(set(paths)):raise ValueError('Duplicate episode paths can leak across splits')
        self.splits=json.loads(Path(split_file).read_text()) if split_file else episode_split(episodes,validation_fraction,split_seed)
        known={e['episode_id'] for e in episodes}
        tr,va=set(self.splits['train']),set(self.splits['validation'])
        if tr & va or tr|va!=known:raise ValueError('Invalid/leaking episode split')
        allowed=known if split=='all' else set(self.splits[split])
        self.episodes=[e for e in episodes if e['episode_id'] in allowed]
        self.horizon=horizon;self.tail=tail;self.cache_limit=cache_episodes;self.cache=OrderedDict();self.samples=[]
        for i,e in enumerate(self.episodes):
            if not e['expert_valid'] or e['schema_version']!=3 or not e['alignment_checked']:raise ValueError('Unvalidated manifest entry')
            for image_index,frame in enumerate(e['rgb_reference_frame_indices']):
                if not 0<=frame<e['state_reference_length']:raise ValueError('RGB frame out of bounds')
                if tail=='drop' and frame+horizon>e['state_reference_length']:continue
                self.samples.append((i,image_index,frame))

    def __len__(self):return len(self.samples)

    def _episode(self,index):
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        entry=self.episodes[index];ep=(self.base/entry['episode_path']).resolve()
        arrays={k:np.load(ep/f'{k}.npy',mmap_mode='r') for k in ['body_q','hand_q','body_ref_q','hand_ref_q','timestamps']}
        with np.load(self.base/entry['rgb_path']) as rgb:
            if rgb['reference_frame'].tolist()!=entry['rgb_reference_frame_indices']:raise ValueError('Manifest/RGB mapping mismatch')
            if not np.array_equal(rgb['timestamps'],arrays['timestamps'][rgb['reference_frame']]):raise ValueError('RGB clock mismatch')
            arrays['head_rgb']=rgb['head_rgb']
        self.cache[index]=arrays
        while len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
        return arrays

    def __getitem__(self,index):
        episode_index,image_index,frame=self.samples[index]
        entry=self.episodes[episode_index];a=self._episode(episode_index)
        length=len(a['timestamps']);indices=np.minimum(np.arange(frame,frame+self.horizon),length-1)
        mask=np.arange(frame,frame+self.horizon)<length
        return {'input':{'head_rgb':a['head_rgb'][image_index].copy(),
                         'body_q':np.array(a['body_q'][frame],dtype=np.float32),
                         'hand_q':np.array(a['hand_q'][frame],dtype=np.float32),
                         'instruction':entry['instruction']},
                'target':{'body_ref_q':np.array(a['body_ref_q'][indices],dtype=np.float32),
                          'hand_ref_q':np.array(a['hand_ref_q'][indices],dtype=np.float32),
                          'action_mask':mask},
                'episode_id':entry['episode_id'],'reference_frame':frame,
                'timestamp':float(a['timestamps'][frame])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=50);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();dataset=VisionActionDataset(args.manifest,horizon=args.horizon,split_seed=args.seed)
    if not len(dataset):raise ValueError('Dataset has no samples')
    (args.manifest.parent/'splits.json').write_text(json.dumps(dataset.splits,indent=2)+'\n')
    first,last=dataset[0],dataset[-1]
    print(json.dumps(dict(episodes=len(dataset.episodes),samples=len(dataset),horizon=args.horizon,
        rgb_shape=list(first['input']['head_rgb'].shape),body_action_shape=list(first['target']['body_ref_q'].shape),
        hand_action_shape=list(first['target']['hand_ref_q'].shape),last_valid_action_frames=int(last['target']['action_mask'].sum()),splits=dataset.splits),indent=2))


if __name__=='__main__':main()
