# GNNcpuonly

CPU-only, inference-only installer and command-line tool for the **Greenleaf-lab graph neural network for DNA
hairpin melting** (Ke et al., *Nat Commun* 2025). Give it DNA sequences; it folds them with NUPACK and predicts

- **dH** (enthalpy of folding, kcal/mol),
- **Tm** (melting temperature, degC),
- **dG37** (free energy at 37 degC, kcal/mol; derived from dH and Tm).

This is an unofficial wrapper. The model, training code and data are the work of the authors of the paper
(see [NOTICE.md](NOTICE.md) for credits, licences and how to cite).

## Quick start

```bash
git clone <this repository> GNNcpuonly && cd GNNcpuonly
./install.sh --accept-nupack-license           # 2-10 min, about 2 GB, no GPU, no root, no conda
bash example/run_example.sh ~/gnn_example      # small test; prints PASS when results match example/expected_output
```
Then predict your own sequences (files of any mix of CSV / FASTA / text in a directory):

```bash
# with Slurm
sbatch /path/to/GNNcpuonly/run_gnn_infer_cpu.slabmk  my_sequences/  my_results/
# without Slurm
/path/to/GNNcpuonly/venv/bin/python /path/to/GNNcpuonly/gnn_predict.py  my_sequences/  -o my_results/
```
`install.sh` prints the exact commands for your paths. See [INSTALL.md](INSTALL.md) (options, what the installer does,
troubleshooting) and [USAGE.md](USAGE.md) (input formats, output columns, options, caveats).

## What it does

```
sequence --NUPACK MFE (dna, 37 C, 1 M Na+)--> dot-bracket --> graph --> GNN --> dH, Tm --> dG37
```
Input files can also carry your own dot-bracket structure, which skips the folding step.

## Speed

27,730 hairpins (all `arr` sequences of the training data), 8 CPU threads, one cluster node, folding included:

| step | time |
|---|---|
| NUPACK folding | 3.6 s |
| graph building | 6.2 s |
| GNN forward pass | 4.2 s |
| **total** (start-up and writing included) | **15.1 s** (about 1800 sequences/s) |

A GPU does not help end to end, since folding and graph building dominate; the CPU build is why this repo is 1.7 GB
instead of about 6 GB. Predictions agree with the CUDA build to within 0.010 kcal/mol (dH) and 0.009 degC (Tm).

## Things to know

- The network was trained on **short hairpins, 11-24 nt** (Tm about 14-69 degC). Other lengths still get a
  prediction but are flagged `in_train_length_range=False`: treat them as extrapolation.
- The network sees only sequence and structure; its numbers refer to the Array Melt measurement conditions. The NUPACK
  options (`--celsius`, `--sodium`, `--magnesium`) only change which structure is folded.
- Pinned versions: nnn_paper `802cd63`, torch 1.12.1 (CPU), torch-geometric 2.3.1, NUPACK 4.0.0.23 (Python 3.9).
  Linux x86_64 only.

## Repository layout

| file | purpose |
|---|---|
| `install.sh` | one-step installer (downloads and verifies everything, builds `venv/`) |
| `gnn_predict.py` | the prediction tool |
| `mk_core.py` | generates `gnn_core.py` (model code) and `train_stats.json` from the pinned nnn_paper clone |
| `run_gnn_infer_cpu.slabmk.in` | Slurm job template (filled in by `install.sh`) |
| `example/` | small input, expected output, `run_example.sh`, `compare_example.py` |
| `NOTICE.md`, `LICENSE` | credits and licences |

Created by `install.sh` and not part of git: `venv/`, `nnn_paper/`, `nnn_data/`, `bin/`, `downloads/`,
`third_party/`, `gnn_core.py`, `train_stats.json`, `run_gnn_infer_cpu.slabmk`.

## Licence

The scripts and documents in this repository are MIT licensed ([LICENSE](LICENSE)). Downloaded components keep their
own licences: nnn_paper (MIT), model weights and data (CC-BY-4.0), **NUPACK (non-commercial academic use only)**.
Details in [NOTICE.md](NOTICE.md).
