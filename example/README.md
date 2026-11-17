# Example

Small, ready-to-run test: 4 input files (31 sequences, all supported formats) plus the output a correct run produces.

```
example/
  input/                         what you feed in (one directory, several files)
    hairpins.csv                 name,sequence: 12 hairpins, NUPACK folds them
    hairpins_with_structure.csv  SEQID,RefSeq,TargetStruct,dH_measured,Tm_measured:
                                 12 hairpins with a given structure (no folding)
    longer_seqs.fa               FASTA, 26/30/40 nt (above the training range: flagged)
    mixed.txt                    "name sequence" lines, a comment, one invalid sequence
  expected_output/               reference results
  run_example.sh                 copies the input to your directory and runs it
  compare_example.py             checks your output against expected_output
```

## Run it

```bash
bash example/run_example.sh ~/gnn_example
```
With Slurm (`sbatch` found) this submits `run_gnn_infer_cpu.slabmk` from `~/gnn_example`; wait for the job, then
```bash
venv/bin/python example/compare_example.py ~/gnn_example/output     # prints PASS
```
Without Slurm it runs locally and compares at once. By hand:
```bash
mkdir -p ~/gnn_example && cd ~/gnn_example && cp -r /path/to/GNNcpuonly/example/input .
sbatch /path/to/GNNcpuonly/run_gnn_infer_cpu.slabmk input output
```

## What you should see

`output/` has one `<input file>.gnn_pred.csv` per input file plus `all_predictions.csv` (31 rows):

| input file | rows | notes |
|--------------------------------------------|--------------|----------------------------------------------|
| `hairpins.csv` | 12 ok | structure from NUPACK MFE (`structure_source=nupack_mfe`) |
| `hairpins_with_structure.csv` | 12 ok | structure taken from `TargetStruct` (`structure_source=input`); predictions match the measurements, dH MAE about 3.4, Tm MAE about 1.3 |
| `longer_seqs.fa` | 3 ok | `in_train_length_range=False` (extrapolation), see below |
| `mixed.txt` | 3 ok + 1 skipped | `bad_example` has `status=invalid_sequence(ACGT only, len>=4)` and empty predictions |

`compare_example.py` prints `PASS` when every prediction matches `expected_output` within 0.01.

`hp_gc_rich` (a 15-nt GC-rich hairpin) comes out at Tm of about 78.6 degC, above the 13.6-68.6 degC seen in training, so
even an in-range length can give an extrapolated value.

## The three longer sequences

From `longer_seqs.fa.gnn_pred.csv` (NUPACK MFE at 37 degC, 1 M Na+; longer than the 11-24 nt training range, so
extrapolation):

| name | length | dH (kcal/mol) | Tm (degC) | dG37 (kcal/mol) |
|---|---|---|---|---|
| long_26nt | 26 | -46.6 | 52.8 | -2.26 |
| long_30nt | 30 | -28.9 | 39.1 | -0.20 |
| long_40nt | 40 | -17.6 | 29.8 | 0.42 |

MFE structures used:

```
long_26nt ((((((((((......))))))))))
long_30nt ....((((((((((......))))))))))
long_40nt ....(((((((((((..............)))))))))))
```
