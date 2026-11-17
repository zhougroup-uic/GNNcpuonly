# Installation

## Requirements

- Linux x86_64 (the pinned wheels are manylinux x86_64), bash, `git`, `curl`, `tar`, `python3` (any 3.x, only used for
  unzipping and checksums)
- internet access during installation
- about 2 GB disk and 2-10 minutes; no root, no conda, no GPU
- Python 3.9 for the environment: taken from the system if present, otherwise downloaded by
  [uv](https://github.com/astral-sh/uv). (The NUPACK wheel is built for Python 3.9, so do not change it.)

## Install

```bash
git clone <this repository> GNNcpuonly
cd GNNcpuonly
./install.sh --accept-nupack-license
```

**Read the NUPACK licence first.** NUPACK is free for non-commercial academic use only. The installer downloads the
NUPACK wheel from a public GitHub mirror (`rwollman/NUPACK`), which redistributes it under that licence. The licence text
is saved to `third_party/NUPACK_LICENSE.txt`. Without `--accept-nupack-license` the installer shows it and asks
(or stops, when not run interactively). If you would rather use a wheel you downloaded yourself from
[nupack.org](https://www.nupack.org) (free academic account; `nupack-4.0.0.23-cp39-cp39-manylinux2014_x86_64.whl`),
pass `--wheel-file /path/to/it.whl`; the checksum must match the pinned build.

On a cluster, run the installer on a compute node if your site does not allow builds on the login node:

```bash
srun -c 4 --mem 8G -t 30 ./install.sh --accept-nupack-license --partition mypartition
```

### Options

| option | meaning |
|---|---|
| `--accept-nupack-license` | accept the NUPACK licence non-interactively |
| `--wheel-file FILE` | use this local NUPACK wheel instead of downloading it |
| `--partition NAME`, `--account NAME` | Slurm settings written into `run_gnn_infer_cpu.slabmk` |
| `--cpus N`, `--mem SIZE`, `--time HH:MM:SS` | job resources written into the job script (default 8, 16G, 2:00:00) |
| `--no-test` | skip the final smoke test (about 10 s) |
| `--force` | rebuild the venv and download everything again |
| `--keep-downloads` | keep `downloads/` (the 108 MB Zenodo zip) |
| `-h` | help |

Environment: `UV_CACHE_DIR` (uv's download cache, default `~/.cache/uv`; on a network file system point it to local or
scratch space), `UV_LINK_MODE` (default `copy`).

The installer is safe to re-run; finished steps are skipped (a second run takes seconds).

## What the installer does

1. downloads `uv` (0.13.9) into `bin/`, unless `uv` is already on your PATH
2. clones [GreenleafLab/nnn_paper](https://github.com/GreenleafLab/nnn_paper) into `nnn_paper/` at commit `802cd63`
3. downloads the trained weights from Zenodo (record 10.5281/zenodo.15299578), checks the zip's MD5 and the weights'
   SHA-256, and extracts `nnn_data/models/gnn_state_dict_ancient-sound-259.pt`
4. downloads the NUPACK wheel (SHA-256 checked)
5. builds `venv/` with uv: torch 1.12.1 (CPU wheel), torch-geometric 2.3.1 (without its scipy / scikit-learn
   dependencies), numpy 1.23.1, pandas 1.5.1, NUPACK 4.0.0.23; then deletes the C++ headers and tests shipped inside the torch
   wheel (about 1 GB, not needed at run time)
6. runs `mk_core.py`: writes `gnn_core.py` (the model code, extracted from `nnn_paper/nnn/gnn.py`, no heavy imports) and
   `train_stats.json` (training-set statistics used to scale the outputs)
7. writes `run_gnn_infer_cpu.slabmk` from the template with your paths and Slurm options
8. runs the example locally and checks it against `example/expected_output`

Everything lands inside the repository directory and is git-ignored. To uninstall, delete the directory (and, if you
wish, uv's cache).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `NUPACK licence not accepted` | read `third_party/NUPACK_LICENSE.txt`, re-run with `--accept-nupack-license` |
| `checksum mismatch` | a download was corrupted or upstream changed; re-run with `--force`; do not skip the check |
| `Linux x86_64 only` | the pinned wheels are manylinux x86_64; other platforms are not supported by this repo |
| `ModuleNotFoundError` for a package in `~/.local` or `PYTHONPATH` | the scripts set `PYTHONNOUSERSITE=1`; use the venv python as shown, not the system one |
| very slow install on a network file system | set `UV_CACHE_DIR` to a local/scratch path |
| smoke test differs from `expected_output` | tolerance is 0.01; a bigger difference means a wrong package version: re-run with `--force` |
