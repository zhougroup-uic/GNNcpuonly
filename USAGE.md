# Usage

Predict **dH (kcal/mol), Tm (degC) and dG37 (kcal/mol)** for every DNA sequence in a directory. Each sequence is folded
with NUPACK (minimum-free-energy structure), turned into a graph, and scored by the saved GNN.

## Quick start

```bash
# with Slurm (writes slurm-<id>.out and the results next to where you submit)
cd ~/my_work
sbatch /path/to/GNNcpuonly/run_gnn_infer_cpu.slabmk  my_sequences/  [my_results/]
# without Slurm
/path/to/GNNcpuonly/venv/bin/python /path/to/GNNcpuonly/gnn_predict.py  my_sequences/  [-o my_results/]
```
Default output directory: `./gnn_pred`. `install.sh` prints these commands with your real paths.

## Try the example first

```bash
bash /path/to/GNNcpuonly/example/run_example.sh ~/gnn_example
```
It copies `example/input/` (4 files, 31 sequences, in each input format), runs the inference (via Slurm if
`sbatch` exists, otherwise locally) and, when finished, `compare_example.py` prints `PASS` if the results match
`example/expected_output` within 0.01. See [example/README.md](example/README.md).

## Command line

```
gnn_predict.py INDIR [-o OUTDIR] [--celsius T] [--sodium M] [--magnesium M] [--batch-size N] [--threads N] [--weights FILE]
```

| option | meaning |
|---|---|
| `INDIR` | directory with the sequence files (required, not searched recursively) |
| `-o`, `--outdir` | output directory, default `./gnn_pred` |
| `--celsius T` | NUPACK folding temperature, default 37 |
| `--sodium M` | NUPACK [Na+] in mol/L, default 1.0 |
| `--magnesium M` | NUPACK [Mg2+] in mol/L, default 0 |
| `--threads N` | CPU processes/threads, default `SLURM_CPUS_PER_TASK`, else all CPUs available to the process |
| `--batch-size N` | GNN batch size, default 1024 |
| `--weights FILE` | alternative state-dict `.pt` (default: the paper's `ancient-sound-259`) |

The Slurm script takes `INPUT_DIR [OUTPUT_DIR] [options]`; omit `OUTPUT_DIR` if your next argument starts with `-`,
e.g. `sbatch run_gnn_infer_cpu.slabmk seqs --celsius 25 --sodium 0.15`.

## Input files

Every file in `INDIR` with one of these extensions is processed, each file separately:

**CSV / TSV** (`.csv`, `.tsv`), header row required:
```
name,sequence
hp1,CAACCAGAAATGGTTG
hp2,AGAAACCGAAAGGCTTCT
```
- sequence column: `sequence`, `seq`, `RefSeq` (or `Sequence`)
- name column (optional): `name`, `id`, `SEQID` (otherwise `<file>_<row>`)
- structure column (optional): `structure`, `TargetStruct`, `dotbracket`. If present, that dot-bracket string is used
  and the NUPACK folding is skipped (e.g. `((((((....))))))`).

**FASTA** (`.fa`, `.fasta`, `.fna`): the first word after `>` is the name.

**Plain text** (`.txt`, `.seq`): one sequence per line, or `name sequence`; lines starting with `#` are ignored.

Sequences are upper-cased and `U` is read as `T`. Only A/C/G/T are accepted, with at least 4 nt.

## Output

For each input file `X.ext`: `OUTDIR/X.ext.gnn_pred.csv`, plus `OUTDIR/all_predictions.csv` with everything (extra
first column `file`).

| column | meaning |
|---|---|
| `name`, `sequence` | as read |
| `structure` | dot-bracket used for the prediction |
| `structure_source` | `nupack_mfe` or `input` |
| `nupack_mfe_energy` | NUPACK MFE (kcal/mol), when folded here |
| `length` | nt |
| `dH_pred_kcal_mol` | predicted enthalpy of folding (negative) |
| `Tm_pred_C` | predicted melting temperature |
| `dG37_pred_kcal_mol` | `dH * (1 - 310.15 / (Tm + 273.15))`, two-state free energy at 37 degC |
| `in_train_length_range` | `True` if the length is within the 11-24 nt range the model was trained on |
| `status` | `ok`, `invalid_sequence(ACGT only, len>=4)` or `bad_structure` (prediction columns empty) |

A `Timing (s): ...` line at the end of the log reports the time per stage.

## Things to know

- **Domain.** The model was trained on short hairpins (11-24 nt, Array Melt data, Tm about 14-69 degC). Other
  sequences are still predicted but flagged; values outside the training range are extrapolation, also a short sequence can come out above the
  highest training Tm.
- **Conditions.** The GNN sees only sequence and structure; its numbers correspond to the Array Melt measurement
  conditions. `--celsius/--sodium/--magnesium` only change which structure NUPACK picks.
- **Structure matters.** The prediction follows the MFE structure. If you intend a different structure (or no
  hairpin at all), give it in a `structure` column.
- **Accuracy check.** On 200 held-out hairpins with known structures this pipeline gave dH MAE 2.62 (R2 0.90) and
  Tm MAE 1.73 (R2 0.94) against measurements.
- **Speed.** About 1800 sequences/s on 8 CPUs, folding included (27,730 hairpins in 15 s); the job time limit
  defaults to 2 h.

## Troubleshooting

| message | fix |
|---|---|
| `usage: sbatch run_gnn_infer_cpu.slabmk INPUT_DIR ...` | no input directory given |
| `ERROR: ... is not a directory` | pass a directory, not a file |
| `No input files ([...]) in ...` | no file with a supported extension in the directory |
| `no sequence column` | CSV/TSV needs a `sequence`, `seq` or `RefSeq` header |
| `train_stats.json not found: run ./install.sh first` | the installation did not finish |
| rows with `invalid_sequence...` | non-ACGT characters or fewer than 4 nt |
