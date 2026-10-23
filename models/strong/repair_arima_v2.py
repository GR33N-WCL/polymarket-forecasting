"""Recover failed windows only; reject invalid numerical candidates in root tests."""
from pathlib import Path
import hashlib, json, time
import numpy as np
import pmdarima.arima._auto_solvers as solvers
import repair_arima as original

R = Path(__file__).resolve().parent
root_test = solvers._root_test
rejected = []

def finite_root_test(model, ic, trace):
    try:
        return root_test(model, ic, trace)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        # A nonfinite candidate cannot be considered an AICc winner. The library
        # already assigns infinity to candidates failing its stability test.
        rejected.append({'order': list(model.order), 'reason': str(exc)})
        return float('inf')

solvers._root_test = finite_root_test
if __name__ == '__main__':
    original.main()
    audit = json.loads((R/'results/AutoARIMA/REPAIR_AUDIT.json').read_text())
    audit['invalid_root_candidates_rejected'] = rejected
    audit['recovery_entry'] = 'repair_arima_v2.py'
    audit['root_guard_scope'] = 'Failed windows only. Valid primary predictions are protected byte for byte.'
    (R/'results/AutoARIMA/REPAIR_AUDIT.json').write_text(json.dumps(audit, indent=2))
    print('RECOVERY_ROOT_GUARD', len(rejected), flush=True)
