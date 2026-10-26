"""Explicit versions and classified source fingerprints, independent of output directory names."""
import ast
import hashlib
import json
import warnings
from .common import ROOT,digest,read,require

COMPILER_VERSION='canonical-va-2.0.0'
RUNTIME_VERSION='sonic-native-dds-tabletop-1.1'
POLICY=ROOT/'canonical_va/source_policy.json'


def symbol_hash(path,symbol):
    tree=ast.parse(path.read_text());node=tree
    if symbol=='__module_bindings__':
        nodes=[x for x in tree.body if not isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
        return hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    for part in symbol.split('.'):
        node=next(x for x in node.body if getattr(x,'name',None)==part)
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def canonical_hash(values):
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check_policy(root=ROOT,policy=None,emit=True):
    policy=policy or read(POLICY);actual={};critical={};changes=[];errors=[]
    for rel,entry in policy['sources'].items():
        path=root/rel
        if not path.is_file():
            errors.append('Missing source: '+rel);continue
        actual[rel]=digest(path)
        if entry['classification']=='critical':
            critical[rel]=actual[rel]
            if actual[rel]!=entry['sha256']:errors.append('Runtime-critical source changed: '+rel)
        elif entry['classification']=='mixed':
            for symbol,expected in entry['critical_symbols'].items():
                try:value=symbol_hash(path,symbol)
                except Exception:value=None
                critical[rel+'::'+symbol]=value
                if value!=expected:errors.append('Runtime-critical symbol changed: '+rel+'::'+symbol)
            if actual[rel]!=entry['sha256']:changes.append(dict(path=rel,classification='non-critical portion of mixed module',expected=entry['sha256'],actual=actual[rel]))
        elif actual[rel]!=entry['sha256']:
            changes.append(dict(path=rel,classification='non-critical',expected=entry['sha256'],actual=actual[rel]))
    if emit:
        for change in changes:warnings.warn('Non-critical source changed: '+change['path'],RuntimeWarning)
    require(not errors,'; '.join(errors))
    return dict(compiler_version=COMPILER_VERSION,canonical_runtime_version=RUNTIME_VERSION,
        canonical_runtime_hash=canonical_hash(critical),source_hashes=actual,source_warnings=changes,
        code_fingerprint=canonical_hash(actual),source_policy_sha256=digest(POLICY))


def recorded_sources(folder,provenance):
    """Current runtime critical gate plus immutable recording-source checks; helper drift warns."""
    version=check_policy(emit=False);policy=read(POLICY)
    for rel,h in provenance['source_hashes'].items():
        current=digest(ROOT/rel)
        if current==h:continue
        entry=policy['sources'].get(rel,{})
        require(entry.get('classification') in ('non-critical','mixed'),'Recorded critical runtime changed: '+rel)
        warnings.warn('Recorded helper/evaluator source differs: '+rel,RuntimeWarning)
    for rel,h in provenance['implementation_hashes'].items():
        archived=folder/'recording_source'/rel
        require(digest(archived if archived.is_file() else ROOT/rel)==h,'Changed recording-time implementation '+rel)
    return version
