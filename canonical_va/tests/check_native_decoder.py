"""Read-only port equivalence against existing native release logs; no new replay."""
import numpy as np
from canonical_va.common import ROOT, dump
from canonical_va.policy import Policy, infer
from canonical_va.schema import validate


def main():
    a,_,_=validate(ROOT/'canonical_va/pilot_reference/pilot_0000'); policy=Policy(a)
    folder=ROOT/'kimodo-t6/g1-integration/task8_development_20261007_v3/accepted_touch_allowed_profile/qualification/policy_logs'
    logs={k:np.loadtxt(folder/(k+'.csv'),delimiter=',',skiprows=1,ndmin=2)[:,5:] for k in ('q','dq','base_quat','base_ang_vel','action','token_state')}
    errors=[]
    for j in range(len(logs['action'])-1):
        policy.previous=logs['action'][j]
        dx,_=policy.decoder_input(logs['token_state'][j].astype(np.float32),logs['q'][j],logs['dq'][j],logs['base_quat'][j],logs['base_ang_vel'][j])
        got=infer(policy.decoder,dx)
        errors.append(float(np.max(abs(got-logs['action'][j+1]))))
    report=dict(purpose='read-only historical decoder port equivalence; not new replay evidence',total=len(errors),
                matched=sum(x<1e-4 for x in errors),max_absolute_error=max(errors),atol=1e-4,
                legacy_log_semantic='Historical action row j+1 holds output from decoder input/token row j. New recorder stores current output immediately.')
    dump(ROOT/'canonical_va/native_decoder_equivalence.json',report)
    print(report)
    if report['matched']!=report['total']: raise SystemExit(2)

if __name__=='__main__': main()
