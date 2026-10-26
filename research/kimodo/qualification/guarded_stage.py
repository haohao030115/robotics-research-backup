"""Kernel-enforced read allowlist inherited by Python and native SONIC children."""
import ctypes,os,sys,json,runpy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
libc=ctypes.CDLL(None,use_errno=True)
class Ruleset(ctypes.Structure):_fields_=[('handled',ctypes.c_uint64)]
class Beneath(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed',ctypes.c_uint64),('parent',ctypes.c_int)]

def restrict(paths):
    abi=libc.syscall(444,0,0,1)
    if abi<1:raise RuntimeError('Landlock unavailable; fail closed')
    attr=Ruleset(7) # execute/read_file/read_dir; no write outside handled read policy
    fd=libc.syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
    if fd<0:raise OSError(ctypes.get_errno(),'create Landlock')
    for p in paths:
        p=Path(p).resolve()
        if not p.exists():continue
        parent=os.open(p,os.O_PATH|os.O_CLOEXEC)
        rule=Beneath(7 if p.is_dir() else 5,parent)
        rc=libc.syscall(445,fd,1,ctypes.byref(rule),0);os.close(parent)
        if rc<0:raise OSError(ctypes.get_errno(),'add Landlock '+str(p))
    if libc.prctl(38,1,0,0,0):raise OSError(ctypes.get_errno(),'no_new_privs')
    if libc.syscall(446,fd,0):raise OSError(ctypes.get_errno(),'restrict Landlock')
    os.close(fd);return abi

def main():
    config=json.loads((ROOT/'config/frozen_config.json').read_text())
    script=Path(sys.argv[1]).resolve();args=sys.argv[2:]
    log=ROOT/'qualification'/('access_'+script.stem+'_'+str(os.getpid())+'.jsonl')
    f=log.open('x')
    roots=[str(Path(p).resolve()) for p in config['read_allowlist']]
    sys.path[:]=[p for p in sys.path if p and any(str(Path(p).resolve())==q or str(Path(p).resolve()).startswith(q+'/') for q in roots)]
    abi=restrict(config['read_allowlist'])
    f.write(json.dumps(dict(event='kernel_policy',abi=abi,read_allowlist=config['read_allowlist'],pid=os.getpid()))+'\n');f.flush()
    busy=False
    def audit(event,items):
        nonlocal busy
        if event!='open' or busy or f.closed:return
        busy=True
        try:
            value=items[0]
            if isinstance(value,(str,bytes)):
                path=os.path.abspath(os.fsdecode(value))
                lowered=path.lower()
                prohibited=('/ardy/' in lowered or '/ardy_handoff' in lowered or '/grasp-e2e-20261007-v1/' in lowered or '/artifacts/development_demo/' in lowered)
                f.write(json.dumps(dict(event='open',path=path,mode=str(items[1]),forbidden=prohibited))+'\n')
                if prohibited:f.flush();raise PermissionError('Forbidden qualification input '+path)
        finally:busy=False
    sys.addaudithook(audit)
    sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/canonical/scripts')]
    sys.argv=[str(script),*args]
    try:runpy.run_path(str(script),run_name='__main__')
    finally:f.flush();f.close()
if __name__=='__main__':main()
