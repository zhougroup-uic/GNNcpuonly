#!/usr/bin/env bash
# GNNcpuonly installer: CPU-only, inference-only Greenleaf-lab GNN for DNA hairpin melting.
#
#   git clone <this repo> && cd GNNcpuonly
#   ./install.sh --accept-nupack-license [options]
#
# Everything is installed inside this directory (git-ignored): venv/, nnn_paper/, nnn_data/, bin/, third_party/.
# Needs: Linux x86_64, bash, git, curl, python3 (any 3.x, for unzip/checksums), internet access. Python 3.9 for
# the venv is taken from the system or downloaded by uv. About 2 GB of disk and 2-10 minutes.
# On a cluster, run it on a compute node if your site does not allow builds on the login node, e.g.
#   srun -c 4 --mem 8G -t 30 ./install.sh --accept-nupack-license
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- pinned inputs (everything downloaded is verified against these) ------------------------------------------
NNN_URL=https://github.com/GreenleafLab/nnn_paper.git
NNN_COMMIT=802cd6376400f2a0cfcfc3381c055c3456591e81                  # MIT licence
ZENODO_URL=https://zenodo.org/api/records/15299579/files/SuppMaterials.zip/content      # CC-BY-4.0
ZENODO_MD5=7d3fda964eea0f077193ec9a322de2d8
WEIGHTS_MEMBER=SuppMaterials/SuppFiles/TrainedGNNModel/gnn_state_dict_ancient-sound-259.pt
WEIGHTS_SHA256=573f503261f71322cfb056239538e0b46c11ac3018b8cede9231c4bc4d03c0a3
NUPACK_WHEEL=nupack-4.0.0.23-cp39-cp39-manylinux2014_x86_64.whl
NUPACK_URL=https://github.com/rwollman/NUPACK/raw/master/src/package/$NUPACK_WHEEL
NUPACK_LICENSE_URL=https://raw.githubusercontent.com/rwollman/NUPACK/master/LICENSE
NUPACK_SHA256=853f9d24d6d9e7f8248a25cc9b2e566607d258b3b6148d8e4899265a81615a0b
UV_VERSION=0.13.9
UV_URL=https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz
TORCH_FIND_LINKS=https://download.pytorch.org/whl/torch_stable.html

# ---- options ---------------------------------------------------------------------------------------------------
PYVER=3.9; PARTITION=""; ACCOUNT=""; WALL="2:00:00"; CPUS=8; MEM="16G"
NO_TEST=0; FORCE=0; ACCEPT=0; WHEEL_FILE=""; KEEP=0
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'

Options:
  --accept-nupack-license   required (or answer the prompt): NUPACK is licensed for non-commercial academic use
  --wheel-file FILE         use this local NUPACK wheel (cp39 manylinux x86_64) instead of downloading it from GitHub
  --partition NAME          Slurm partition written into run_gnn_infer_cpu.slabmk (default: cluster default)
  --account NAME            Slurm account written into run_gnn_infer_cpu.slabmk
  --cpus N  --mem SIZE  --time HH:MM:SS     Slurm resources in the job script (default 8, 16G, 2:00:00)
  --python X.Y              venv python (default 3.9; the NUPACK wheel is cp39, so keep it)
  --no-test                 skip the final smoke test (runs the example locally, about 10 s)
  --force                   rebuild venv and re-download everything
  --keep-downloads          keep downloads/ (108 MB Zenodo zip)
Environment: UV_CACHE_DIR (uv download cache; default ~/.cache/uv), UV_LINK_MODE (default copy)
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --accept-nupack-license) ACCEPT=1 ;;
    --wheel-file) WHEEL_FILE=$(realpath "$2"); shift ;;
    --partition) PARTITION=$2; shift ;;
    --account) ACCOUNT=$2; shift ;;
    --cpus) CPUS=$2; shift ;;
    --mem) MEM=$2; shift ;;
    --time) WALL=$2; shift ;;
    --python) PYVER=$2; shift ;;
    --no-test) NO_TEST=1 ;;
    --force) FORCE=1 ;;
    --keep-downloads) KEEP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac; shift
done

say()  { printf '\n==> %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
dl()   { curl -fL --retry 3 --retry-delay 2 -sS -o "$2" "$1" || die "download failed: $1"; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "Linux x86_64 only (the pinned wheels are manylinux x86_64)"
for c in git curl python3 tar sha256sum md5sum; do need "$c"; done
cd "$HERE"
mkdir -p downloads third_party bin nnn_data/models

# ---- NUPACK licence -------------------------------------------------------------------------------------------
# The wheel is downloaded from a GitHub mirror that distributes it under the NUPACK "non-commercial academic use and
# redistribution" licence, so the user has to accept that licence first. (With --wheel-file the user brings a wheel
# obtained from nupack.org under their own account and nothing is asked.)
if [ ! -s third_party/NUPACK_LICENSE.txt ] || [ "$FORCE" = 1 ]; then dl "$NUPACK_LICENSE_URL" third_party/NUPACK_LICENSE.txt; fi
if [ -z "$WHEEL_FILE" ] && [ "$ACCEPT" = 0 ]; then
  say "NUPACK licence (full text in third_party/NUPACK_LICENSE.txt)"
  head -c 900 third_party/NUPACK_LICENSE.txt; echo; echo "[...]"
  if [ -t 0 ]; then
    read -r -p "Do you accept this licence (non-commercial academic use only)? [y/N] " ans
    [ "$ans" = y ] || [ "$ans" = Y ] || die "licence not accepted"
  else
    die "NUPACK licence not accepted: read third_party/NUPACK_LICENSE.txt, then re-run with --accept-nupack-license"
  fi
fi

# ---- uv -------------------------------------------------------------------------------------------------------
say "uv"
if [ -x bin/uv ]; then UV=$HERE/bin/uv
elif command -v uv >/dev/null 2>&1; then UV=$(command -v uv)
else
  dl "$UV_URL" downloads/uv.tar.gz
  tar -xzf downloads/uv.tar.gz --strip-components=1 -C bin
  UV=$HERE/bin/uv
fi
"$UV" --version

# ---- nnn_paper (pinned commit) --------------------------------------------------------------------------------
say "nnn_paper @ ${NNN_COMMIT:0:10}"
[ "$FORCE" = 1 ] && rm -rf nnn_paper
if [ ! -d nnn_paper/.git ]; then git clone --quiet "$NNN_URL" nnn_paper; fi
git -C nnn_paper checkout --quiet --detach "$NNN_COMMIT"
[ "$(git -C nnn_paper rev-parse HEAD)" = "$NNN_COMMIT" ] || die "nnn_paper is not at the pinned commit"

# ---- trained weights (Zenodo, CC-BY-4.0) ----------------------------------------------------------------------
say "trained weights (Zenodo 10.5281/zenodo.15299578)"
PT=nnn_data/models/gnn_state_dict_ancient-sound-259.pt
if [ "$FORCE" = 1 ] || [ ! -s "$PT" ] || [ "$(sha256sum "$PT" | cut -d' ' -f1)" != "$WEIGHTS_SHA256" ]; then
  dl "$ZENODO_URL" downloads/SuppMaterials.zip
  [ "$(md5sum downloads/SuppMaterials.zip | cut -d' ' -f1)" = "$ZENODO_MD5" ] || die "Zenodo zip checksum mismatch"
  python3 -I - "$WEIGHTS_MEMBER" "$PT" <<'PY'
import sys, zipfile
member, dest = sys.argv[1:3]
with zipfile.ZipFile("downloads/SuppMaterials.zip") as z, open(dest, "wb") as f:
    f.write(z.read(member))
PY
  [ "$KEEP" = 1 ] || rm -f downloads/SuppMaterials.zip
fi
[ "$(sha256sum "$PT" | cut -d' ' -f1)" = "$WEIGHTS_SHA256" ] || die "weights checksum mismatch"

# ---- NUPACK wheel (GitHub mirror) -----------------------------------------------------------------------------
say "NUPACK 4 wheel"
if [ -n "$WHEEL_FILE" ]; then WHEEL=$WHEEL_FILE
else
  WHEEL=$HERE/downloads/$NUPACK_WHEEL
  if [ "$FORCE" = 1 ] || [ ! -s "$WHEEL" ]; then dl "$NUPACK_URL" "$WHEEL"; fi
fi
[ "$(sha256sum "$WHEEL" | cut -d' ' -f1)" = "$NUPACK_SHA256" ] || die "NUPACK wheel checksum mismatch ($WHEEL)"

# ---- venv -----------------------------------------------------------------------------------------------------
say "Python $PYVER venv (torch CPU, PyG, nupack)"
export PYTHONNOUSERSITE=1; unset PYTHONUSERBASE PYTHONPATH             # ~/.local packages must not leak in
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
[ "$FORCE" = 1 ] && rm -rf venv
if [ ! -x venv/bin/python ]; then "$UV" venv --python "$PYVER" venv; fi
I=("$UV" pip install --python "$HERE/venv/bin/python")
"${I[@]}" torch==1.12.1+cpu --find-links "$TORCH_FIND_LINKS"
"${I[@]}" --no-deps torch-geometric==2.3.1                   # without scipy/scikit-learn: inference does not use them
"${I[@]}" numpy==1.23.1 pandas==1.5.1 tqdm jinja2 requests pyparsing psutil
"${I[@]}" "$WHEEL"
SP=$(venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
rm -rf "$SP/torch/include" "$SP/torch/test"                  # build-time only files in the torch wheel (about 1 GB)

# ---- slim core + training statistics (from the pinned nnn_paper) ----------------------------------------------
say "gnn_core.py, train_stats.json"
venv/bin/python mk_core.py --clone nnn_paper --out .

# ---- Slurm job script from template ---------------------------------------------------------------------------
say "run_gnn_infer_cpu.slabmk"
EXTRA=""
[ -n "$PARTITION" ] && EXTRA+="#SBATCH --partition=$PARTITION"$'\n'
[ -n "$ACCOUNT" ]   && EXTRA+="#SBATCH --account=$ACCOUNT"$'\n'
T=$(<run_gnn_infer_cpu.slabmk.in)
T=${T//'#@SBATCH_EXTRA@'$'\n'/$EXTRA}
T=${T//@PREFIX@/$HERE}; T=${T//@CPUS@/$CPUS}; T=${T//@MEM@/$MEM}; T=${T//@TIME@/$WALL}
printf '%s\n' "$T" > run_gnn_infer_cpu.slabmk; chmod 755 run_gnn_infer_cpu.slabmk
echo "wrote run_gnn_infer_cpu.slabmk (partition: ${PARTITION:-cluster default}, ${CPUS} CPUs, ${MEM}, ${WALL})"

# ---- smoke test -----------------------------------------------------------------------------------------------
if [ "$NO_TEST" = 0 ]; then
  say "smoke test (example, runs locally)"
  OUT=$(mktemp -d)
  venv/bin/python gnn_predict.py example/input -o "$OUT/output" --threads 4 | grep -E "Timing|Done"
  venv/bin/python example/compare_example.py "$OUT/output" | tail -1
  rm -rf "$OUT"
fi

du -sh venv | sed 's/^/venv size: /'
cat <<EOF

Installed in $HERE
  Slurm:   mkdir -p ~/work && cd ~/work && sbatch $HERE/run_gnn_infer_cpu.slabmk INPUT_DIR [OUTPUT_DIR]
  no Slurm: $HERE/venv/bin/python $HERE/gnn_predict.py INPUT_DIR -o OUTPUT_DIR
  try:     bash $HERE/example/run_example.sh ~/gnn_example
EOF
