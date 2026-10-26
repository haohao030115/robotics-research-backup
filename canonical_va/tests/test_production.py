import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from canonical_va.common import dump, read, digest, jsonl
from canonical_va.versioning import check_policy, symbol_hash
from canonical_va.production import metadata_fields, qa_decision, mark_qa, accepted_episode, pack_samples, training_success
from canonical_va.rendering import renderer_allowed
from canonical_va.aggregate import aggregate
from canonical_va.batch import run_batch, REQUIRED


class ProductionContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def episode(self,name,eid='episode',dtype=np.float32,accepted=True):
        p=self.root/name;p.mkdir();d=p/'va_dataset';d.mkdir();(d/'rgb').mkdir()
        Image.fromarray(np.full((240,320,3),42,np.uint8)).save(d/'rgb/0.png')
        np.savez(d/'samples.npz',sample_id=np.asarray([eid+':sample:1']),body_target=np.zeros((1,5,64),dtype),hand_target=np.arange(70,dtype=dtype).reshape(1,5,14))
        jsonl(d/'sample_mapping.jsonl',[dict(episode_id=eid,sample_id=eid+':sample:1',rgb_path='rgb/0.png')])
        before=np.load(d/'samples.npz')['hand_target'].copy();spec=pack_samples(d)
        self.assertTrue(np.array_equal(before,np.load(d/'samples.npz')['hand_target']))
        e=dict(episode_id=eid,producer_type='kimodo',case={'id':'c','block_xy':[.4,-.1]},block_xy=[.4,-.1],task_arm='left',sample_count=1,expert_valid=accepted,path='va_dataset')
        dump(p/'training_manifest.json',{'episodes':[e]});dump(p/'validation_summary.json',dict(expert_valid=accepted,sample_schema=spec,qa={'full_reconstruction_performed':True}))
        dump(p/'provenance.json',{})
        return p
    def test_software_gate_and_explicit_override(self):
        info=dict(render_succeeded=True,GL_RENDERER='llvmpipe (LLVM)')
        with self.assertRaisesRegex(ValueError,'SOFTWARE_RENDERER_REJECTED'):renderer_allowed(info,'production')
        self.assertTrue(renderer_allowed(info,'audit'));self.assertTrue(renderer_allowed(info,'production',True))
        self.assertEqual(renderer_allowed(dict(render_succeeded=True,GL_RENDERER='MTT S4000'),'production'),[])
    def test_mapping_critical_evaluator_warning(self):
        p=self.root/'module.py';p.write_text('VALUE=1\ndef mapping(): return 1\ndef evaluator(): return 1\n')
        policy={'sources':{'module.py':dict(classification='mixed',sha256=digest(p),critical_symbols={s:symbol_hash(p,s) for s in ('mapping','__module_bindings__')})}}
        check_policy(self.root,policy,False)
        p.write_text(p.read_text().replace('evaluator(): return 1','evaluator(): return 2'))
        self.assertEqual(len(check_policy(self.root,policy,False)['source_warnings']),1)
        p.write_text(p.read_text().replace('mapping(): return 1','mapping(): return 2'))
        with self.assertRaisesRegex(ValueError,'Runtime-critical symbol'):check_policy(self.root,policy,False)
    def test_qa_first_change_sampling_and_certificate(self):
        v=dict(code_fingerprint='code',canonical_runtime_hash='runtime');r=dict(GL_VENDOR='v',GL_RENDERER='r',GL_VERSION='1');registry=self.root/'qa.json'
        full,_,key=qa_decision(v,r,'e','source',registry);self.assertTrue(full)
        episode=self.episode('accepted');mark_qa(registry,key,episode)
        self.assertFalse(qa_decision(v,r,'e','source',registry)[0]);self.assertTrue(qa_decision(v,r,'e','source',registry,1.)[0])
        self.assertTrue(qa_decision(dict(v,code_fingerprint='new'),r,'e','source',registry)[0])
        (episode/'validation_summary.json').write_text('{}');self.assertTrue(qa_decision(v,r,'e','source',registry)[0])
    def test_metadata_left_and_bins(self):
        m=dict(episode_id='e',producer_type='ardy',case={'id':'c','block_xy':[.42,-.12],'block_yaw_deg':30},task_arm='left')
        e=metadata_fields(m,2,True,.05);self.assertEqual(e['task_arm'],'left');self.assertEqual(e['position_bin'],'x8_y-3');self.assertEqual(e['block_yaw_deg'],30)
    def test_aggregate_dedup_reject_and_dtype(self):
        a=self.episode('a');self.episode('b');self.episode('c','bad_dtype',np.float64);self.episode('d','failed',accepted=False);self.episode('e','second')
        report=aggregate([self.root],self.root/'merged')
        self.assertEqual(report['accepted_episodes'],2);self.assertEqual(len(report['duplicates']),1);self.assertEqual(len(report['rejected']),2)
        z=np.load(self.root/'merged/va_dataset/samples.npz');self.assertEqual(z['hand_target'].shape,(2,5,14));self.assertEqual(z['rgb'].shape,(2,240,320,3))
        with self.assertRaisesRegex(ValueError,'Duplicate episode'):aggregate([a,self.root/'b'],self.root/'strict','error')
    def test_checksum_resume_gate(self):
        p=self.episode('a');accepted_episode(p)
        with (p/'va_dataset/samples.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):accepted_episode(p)

    def test_training_retention_keeps_v4_tables_and_rgb(self):
        p=self.episode('training')
        for rel in ('source_reference/metadata.json','source_reference/provenance.json','source_reference/reference.npz','source_reference/reference_validation.json',
                     'canonical_replay/metadata.json','canonical_replay/provenance.json','canonical_replay/sonic_encoder.npz','canonical_replay/references.npz','canonical_replay/observations.npz',
                     'va_dataset/metadata.json','validation/reference_validation.json','validation/replay_validation.json','validation/temporal_validation.json','validation/dataset_validation.json','validation/renderer.json','REPORT.md'):
            q=p/rel;q.parent.mkdir(parents=True,exist_ok=True)
            if q.suffix=='.npz':np.savez(q,ids=np.arange(2))
            elif q.suffix=='.json':dump(q,{})
            else:q.write_text('ok\n')
        (p/'canonical_replay/debug.log').write_text('remove')
        training_success(p)
        self.assertTrue((p/'canonical_replay/sonic_encoder.npz').is_file())
        self.assertTrue((p/'va_dataset/rgb/0.png').is_file())
        self.assertFalse((p/'canonical_replay/debug.log').exists())
    def test_batch_continues_and_resumes(self):
        refs=self.root/'refs';refs.mkdir();(refs/'a_missing').mkdir();valid=refs/'z_valid';valid.mkdir()
        for name in REQUIRED:dump(valid/name,{'episode_id':'episode'})
        out=self.root/'batch';out.mkdir();episode=self.episode('prepared');episode.rename(out/'z_valid');episode=out/'z_valid'
        dump(episode/'provenance.json',{'reference_source_hashes':{n:digest(valid/n) for n in REQUIRED}})
        with patch('subprocess.run') as child:
            self.assertEqual(run_batch(refs,out,runner=child),2);child.assert_not_called()
        result=read(out/'batch_summary.json');self.assertEqual((result['FAIL'],result['SKIP']),(1,1))
        dump(valid/'metadata.json',{'episode_id':'changed'})
        self.assertEqual(run_batch(refs,out),2);self.assertEqual(read(out/'batch_summary.json')['FAIL'],2)
    def test_production_disallows_recorded_native(self):
        from canonical_va.pipeline import run
        with patch('canonical_va.replay.replay') as native:
            self.assertEqual(run(self.root,self.root/'rejected',mode='production',recorded_episode=self.root),2);native.assert_not_called()
        self.assertIn('requires new native',read(self.root/'rejected/validation_summary.json')['error'])

if __name__=='__main__':unittest.main()
