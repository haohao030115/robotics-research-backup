"""Apply the shared strict gate to a completed native replay summary."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .validation import UnifiedValidationPolicy

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("result",type=Path); p.add_argument("--out",type=Path)
    a=p.parse_args(argv); result=json.loads(a.result.read_text())
    verdict=UnifiedValidationPolicy().evaluate_native_result(result)
    payload={"policy_version":UnifiedValidationPolicy.VERSION,"source":str(a.result.resolve()),
             "result":verdict.as_dict()}
    out=a.out or a.result.with_name("unified_strict_validation.json")
    out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2)); return 0 if verdict.expert_valid else 2
if __name__ == "__main__": raise SystemExit(main())
