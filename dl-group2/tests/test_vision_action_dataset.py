from pathlib import Path
import sys
import tempfile,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vision_action_dataset import VisionActionDataset


def test_alignment_tail_and_split():
    with tempfile.TemporaryDirectory() as tmp:
        b=Path(tmp);entries=[]
        for k in range(3):
            ep=b/f'ep{k}';ep.mkdir();n=13;frames=np.array([0,5,10]);ts=np.arange(n)/50
            np.save(ep/'timestamps.npy',ts)
            for key,width in [('body_q',29),('hand_q',14),('body_ref_q',29),('hand_ref_q',14)]:np.save(ep/f'{key}.npy',np.tile(np.arange(n)[:,None],(1,width)))
            np.savez(ep/'images.npz',head_rgb=np.broadcast_to(frames.astype(np.uint8)[:,None,None,None],(3,240,320,3)).copy(),reference_frame=frames,timestamps=ts[frames])
            entries.append(dict(episode_id=ep.name,episode_path=ep.name,expert_valid=True,schema_version=3,alignment_checked=True,rgb_path=f'{ep.name}/images.npz',rgb_reference_frame_indices=frames.tolist(),state_reference_length=n,instruction='test'))
        p=b/'manifest.json';p.write_text(json.dumps(dict(schema_version=3,episodes=entries)))
        full=VisionActionDataset(p,horizon=6);assert len(full)==9
        sample=full[2];assert sample['input']['head_rgb'][0,0,0]==10;assert sample['reference_frame']==10;assert sample['target']['body_ref_q'][:,0].tolist()==[10,11,12,12,12,12];assert sample['target']['action_mask'].tolist()==[True,True,True,False,False,False]
        assert len(VisionActionDataset(p,horizon=6,tail='drop'))==6
        train=VisionActionDataset(p,split='train');val=VisionActionDataset(p,split='validation')
        assert not {x['episode_id'] for x in train.episodes}&{x['episode_id'] for x in val.episodes}
        assert len(train)+len(val)==len(full)
        duplicate=json.loads(p.read_text());duplicate['episodes'][1]['episode_path']='ep0';bad_manifest=b/'duplicate.json';bad_manifest.write_text(json.dumps(duplicate))
        try:VisionActionDataset(bad_manifest)
        except ValueError:pass
        else:raise AssertionError('Duplicate episode path accepted')
        bad=b/'bad_split.json';bad.write_text(json.dumps({'train':['ep0','ep1'],'validation':['ep1','ep2']}))
        try:VisionActionDataset(p,split_file=bad)
        except ValueError:pass
        else:raise AssertionError('Split leakage accepted')
        with np.load(b/'ep0/images.npz') as z:a=dict(z)
        a['timestamps']=a['timestamps']+.02;np.savez(b/'ep0/images.npz',**a)
        try:VisionActionDataset(p)[0]
        except ValueError:pass
        else:raise AssertionError('Misaligned image accepted')
    print('PASS synthetic chunk alignment, tail padding/mask, drop mode, split leakage and timestamp rejection')

if __name__=='__main__':test_alignment_tail_and_split()
