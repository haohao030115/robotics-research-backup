"""Report both no-table and table-touch diagnostic rules, retaining the original audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from validate_grasp_run import validate


def opposing(row):
    cube = np.array(row['cube_world_pos'])
    thumbs = [c for c in row['contacts'] if 'right_hand_thumb' in c['body'] and c['normal_force_n'] > 1e-4]
    fingers = [c for c in row['contacts'] if any(x in c['body'] for x in ('right_hand_index', 'right_hand_middle')) and c['normal_force_n'] > 1e-4]
    return any(np.dot(a['cube_outward_normal'], b['cube_outward_normal']) < 0 and
               np.dot(np.array(a['position'])-cube, np.array(b['position'])-cube) < 0 for a in thumbs for b in fingers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    if (a.run/'task_validation.json').exists() or (a.run/'validation.json').exists():
        p.error('Preserve existing validation')
    strict = validate(a.run)
    (a.run/'validation.json').write_text(json.dumps(strict, indent=2)+'\n')
    raw = json.loads((a.run/'result.json').read_text())
    rows = json.loads((a.run/'physics_steps.json').read_text())
    current = maximum = 0.
    for r in rows:
        valid = r['playing'] and r['lift_m'] >= .1 and not r['cube_table_contact'] and opposing(r)
        current = current+.005 if valid else 0.
        maximum = max(maximum, current)
    violations = [c for r in rows for c in r['support_contacts'] if c[1] != 'task_table']
    checks = dict(strict['checks'])
    checks.pop('no_support_contacts', None)
    checks.pop('runtime_integrity_passed', None)
    remaining = [x for x in raw['integrity_failure_reasons'] if x != 'robot_environment_support_contact']
    checks['non_table_support_absent'] = not violations
    checks['other_integrity_failures_absent'] = not remaining
    physical = maximum >= 2.-1e-9
    table_steps = sum(any(c[1] == 'task_table' for c in r['support_contacts']) for r in rows)
    out = dict(rule='same geometric lift/hold criterion as Kimodo; robot-table contact diagnostic',
               physical_success=physical, expert_valid=physical and all(checks.values()),
               strict_no_table_expert_valid=bool(physical and strict['expert_valid_confirmed']),
               max_lift_m=raw['max_lift_m'], geometric_opposing_hold_s=maximum,
               robot_table_contact_steps=table_steps, robot_table_contact_duration_s=.005*table_steps,
               checks=checks, classification='development_tuned_ARDY_pipeline; not raw-model zero-shot score')
    (a.run/'task_validation.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))
    return 0 if out['expert_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
