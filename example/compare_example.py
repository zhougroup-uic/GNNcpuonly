#!/usr/bin/env python3
"""Compare a run_gnn_infer output directory with the example's expected_output.
Usage: python3 compare_example.py OUTPUT_DIR      (needs pandas; the install's venv python has it)
"""
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found: run with the install's venv/bin/python")

TOL = 0.01
exp_dir = Path(__file__).resolve().parent / "expected_output"
out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
num = ["nupack_mfe_energy", "dH_pred_kcal_mol", "Tm_pred_C", "dG37_pred_kcal_mol"]
txt = ["file", "name", "sequence", "structure", "structure_source", "length", "in_train_length_range", "status"]

bad = 0
for e in sorted(exp_dir.glob("*.csv")):
    o = out_dir / e.name
    if not o.exists():
        print(f"MISSING  {e.name}"); bad += 1; continue
    a, b = pd.read_csv(e), pd.read_csv(o)
    if a.shape != b.shape:
        print(f"SHAPE    {e.name}: expected {a.shape}, got {b.shape}"); bad += 1; continue
    t_ok = all(a[c].equals(b[c]) for c in txt if c in a.columns)
    d = max(((a[c] - b[c]).abs().max() for c in num if c in a.columns), default=0)
    d = 0 if pd.isna(d) else d
    ok = t_ok and d <= TOL
    bad += not ok
    print(f"{'OK      ' if ok else 'DIFF    '} {e.name:42s} rows={len(a):3d} max|diff|={d:.4f}" + ("" if t_ok else "  (text columns differ)"))
print("PASS" if not bad else f"FAIL ({bad} problem file(s))")
sys.exit(1 if bad else 0)
