#!/bin/bash
# Run the example: copies example/input to WORKDIR and runs the inference there.
# With Slurm (sbatch found) it submits run_gnn_infer_cpu.slabmk; otherwise it runs locally.
# Usage: bash example/run_example.sh [WORKDIR]      (default ./gnn_example)
set -e
G=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W="${1:-gnn_example}"
mkdir -p "$W"
if [ -e "$W/input" ] || [ -e "$W/output" ]; then
  echo "ERROR: $W/input or $W/output already exists; pick another WORKDIR or remove them." >&2; exit 1
fi
cp -r "$G/example/input" "$W/input"
cd "$W"
if command -v sbatch >/dev/null 2>&1 && [ -x "$G/run_gnn_infer_cpu.slabmk" ]; then
  sbatch "$G/run_gnn_infer_cpu.slabmk" input output
  echo "Submitted from $(pwd). When done:  $G/venv/bin/python $G/example/compare_example.py $(pwd)/output"
else
  PYTHONNOUSERSITE=1 "$G/venv/bin/python" "$G/gnn_predict.py" input -o output
  "$G/venv/bin/python" "$G/example/compare_example.py" output
fi
