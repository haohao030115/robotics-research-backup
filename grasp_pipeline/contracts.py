"""Small versioned, checksummed artifact contracts; no ML/runtime imports."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

VERSION = 1
STATES = ('NOT_STARTED','RUNNING','SUCCEEDED','FAILED','BLOCKED','SKIPPED')
STAGES = ('compile','dataset_build','dataset_audit','training')

class PipelineError(ValueError):
    def __init__(self, message, kind='invalid_artifact', code=2):
        super().__init__(message); self.kind=kind; self.code=code

def require(ok, message, kind='invalid_artifact'):
    if not ok: raise PipelineError(message,kind)

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        os.fchmod(fd,0o644)
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def files_record(paths):
    return {str(Path(p).resolve()):sha(p) for p in sorted(set(map(Path,paths)))}

def verify_files(record):
    for p,h in record.items(): require(Path(p).is_file() and sha(p)==h,'Artifact changed or missing: '+p)

def seal(folder, names):
    folder=Path(folder); hashes={n:sha(folder/n) for n in names}
    write(folder/'COMPLETED.json',dict(schema_version=VERSION,files=hashes))
    return files_record([folder/n for n in [*names,'COMPLETED.json']])

def verify_seal(folder):
    folder=Path(folder); v=read(folder/'COMPLETED.json')
    require(v.get('schema_version')==VERSION,'Unsupported completion contract')
    for name,h in v['files'].items():
        p=folder/name;require(p.resolve().is_relative_to(folder.resolve()),'Unsafe sealed artifact path')
        require(p.is_file() and sha(p)==h,'Sealed artifact changed: '+str(p))
    return v

def validate_experiment(m):
    require(m.get('schema_version')==VERSION and m.get('artifact_type')=='experiment','Invalid experiment schema')
    require(bool(m.get('experiment_id')) and isinstance(m.get('config'),dict),'Invalid experiment identity/config')
    require(set(m['stages'])==set(STAGES),'Invalid stage set')
    for name,s in m['stages'].items():
        require(s['state'] in STATES and isinstance(s['attempts'],list),'Invalid stage state: '+name)
    require(m.get('training_authorized') is False,'Phase B training authorization must remain false')
    require(m['stages']['training']['state']=='NOT_STARTED','Phase B cannot authorize or execute training')
    return m
