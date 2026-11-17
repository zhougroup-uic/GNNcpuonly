#!/usr/bin/env python
"""Predict dH, Tm (and derived dG37) for every DNA sequence found in an input directory.

Pipeline per sequence: (optional) NUPACK MFE fold -> dot-bracket -> graph -> saved GNN
(Ke et al. 2025, gnn_state_dict_ancient-sound-259.pt) -> dH [kcal/mol], Tm [degC].
dG37 = dH * (1 - 310.15 / (Tm + 273.15))  (two-state; reproduces dG_37 in the training table).

Input files in INDIR (each handled separately, non-recursive):
  *.csv / *.tsv        header row; sequence column = sequence|seq|RefSeq ; optional name column
                       = name|id|SEQID ; optional structure column = structure|TargetStruct|dotbracket
  *.fa / *.fasta/.fna  FASTA
  *.txt / *.seq        one sequence per line, or "name<space>sequence"; lines starting with # ignored
"""
import os, sys, json, argparse, warnings, time
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings("ignore")
G = Path(os.environ.get("GNNCPU_HOME", Path(__file__).resolve().parent))   # install dir (contains gnn_core.py, nnn_data/)

import numpy as np
import pandas as pd

SEQ_COLS = ["sequence", "seq", "RefSeq", "refseq", "Sequence"]
NAME_COLS = ["name", "id", "SEQID", "seqid", "Name", "ID"]
STRUCT_COLS = ["structure", "TargetStruct", "dotbracket", "Structure"]
EXTS = {".csv", ".tsv", ".fa", ".fasta", ".fna", ".txt", ".seq"}


def read_file(p: Path) -> pd.DataFrame:
    ext = p.suffix.lower()
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="," if ext == ".csv" else "\t", dtype=str)
        sc = next((c for c in SEQ_COLS if c in df.columns), None)
        if sc is None:
            raise ValueError(f"{p.name}: no sequence column (one of {SEQ_COLS}); has {list(df.columns)}")
        nc = next((c for c in NAME_COLS if c in df.columns), None)
        tc = next((c for c in STRUCT_COLS if c in df.columns), None)
        out = pd.DataFrame({"name": df[nc] if nc else [f"{p.stem}_{i+1}" for i in range(len(df))],
                            "sequence": df[sc], "structure": df[tc] if tc else None})
    elif ext in (".fa", ".fasta", ".fna"):
        names, seqs, cur = [], [], None
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                names.append(line[1:].split()[0] if line[1:].split() else f"{p.stem}_{len(names)+1}")
                seqs.append("")
            elif names:
                seqs[-1] += line
        out = pd.DataFrame({"name": names, "sequence": seqs, "structure": None})
    else:
        names, seqs = [], []
        for line in p.read_text().splitlines():
            t = line.split()
            if not t or line.lstrip().startswith("#"):
                continue
            names.append(t[0] if len(t) > 1 else f"{p.stem}_{len(names)+1}")
            seqs.append(t[-1])
        out = pd.DataFrame({"name": names, "sequence": seqs, "structure": None})
    out["sequence"] = out["sequence"].astype(str).str.strip().str.upper().str.replace("U", "T")
    return out


_MODEL_KW = {}


_MODEL = None


def _fold(seq):
    global _MODEL
    import nupack
    if _MODEL is None:                      # build the (slow) NUPACK model once per worker process
        _MODEL = nupack.Model(material="dna", **_MODEL_KW)
    r = nupack.mfe(strands=[seq], model=_MODEL)[0]
    return str(r.structure), float(r.energy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="directory with sequence files")
    ap.add_argument("-o", "--outdir", default="gnn_pred", help="output directory (default ./gnn_pred)")
    ap.add_argument("--celsius", type=float, default=37.0, help="NUPACK folding temperature (default 37)")
    ap.add_argument("--sodium", type=float, default=1.0, help="NUPACK [Na+] in M (default 1.0)")
    ap.add_argument("--magnesium", type=float, default=0.0, help="NUPACK [Mg2+] in M (default 0)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU processes/threads (default: SLURM_CPUS_PER_TASK, else all CPUs available to this process)")
    ap.add_argument("--weights", default=str(G / "nnn_data/models/gnn_state_dict_ancient-sound-259.pt"))
    a = ap.parse_args()
    _MODEL_KW.update(celsius=a.celsius, sodium=a.sodium, magnesium=a.magnesium)

    T0 = time.perf_counter(); tm = {}
    nthreads = a.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or len(os.sched_getaffinity(0))
    indir, outdir = Path(a.indir), Path(a.outdir)
    files = sorted(f for f in indir.iterdir() if f.is_file() and f.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"No input files ({sorted(EXTS)}) in {indir}")
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in files:
        d = read_file(f)
        d.insert(0, "file", f.name)
        print(f"{f.name}: {len(d)} sequences")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    tm["read"] = time.perf_counter() - T0

    # training-set statistics (to turn normalised outputs back into dH / Tm) and training length range
    stats_file = G / "train_stats.json"
    if not stats_file.exists():
        sys.exit(f"{stats_file} not found: run ./install.sh first")
    st = json.load(open(stats_file)); sumstats = st["sumstats"]; lmin, lmax = st["len_min"], st["len_max"]

    ok = df.sequence.str.fullmatch(r"[ACGT]{4,}")
    df["status"] = np.where(ok, "ok", "invalid_sequence(ACGT only, len>=4)")
    df["length"] = df.sequence.str.len()
    df["in_train_length_range"] = df.length.between(lmin, lmax)

    # 1. structures
    t = time.perf_counter()
    need = df.index[ok & df.structure.isna()]
    df["structure_source"] = np.where(df.structure.notna(), "input", "nupack_mfe")
    df["nupack_mfe_energy"] = np.nan
    if len(need):
        uniq = sorted(set(df.loc[need, "sequence"]))
        nproc = nthreads
        print(f"Folding {len(uniq)} unique sequences with NUPACK ({a.celsius} C, Na {a.sodium} M, Mg {a.magnesium} M), {nproc} procs")
        import nupack  # noqa: F401  (import before fork)
        with Pool(min(nproc, len(uniq))) as pool:
            res = dict(zip(uniq, pool.map(_fold, uniq, chunksize=max(1, len(uniq) // (nproc * 8)))))
        df.loc[need, "structure"] = df.loc[need, "sequence"].map(lambda s: res[s][0])
        df.loc[need, "nupack_mfe_energy"] = df.loc[need, "sequence"].map(lambda s: res[s][1])
    bad = ok & ((df.structure.str.len() != df.sequence.str.len()) |
                (df.structure.str.count(r"\(") != df.structure.str.count(r"\)")))
    df.loc[bad, "status"] = "bad_structure"
    run = df.status == "ok"
    tm["fold"] = time.perf_counter() - t

    # 2. GNN
    import torch
    from torch_geometric.loader import DataLoader
    sys.path.insert(0, str(G))
    from gnn_core import GNN, row2graphdata, unorm
    device = torch.device("cpu")
    torch.set_num_threads(nthreads)
    config = dict(hidden_channels=125, pooling="Set2Set", processing_steps=10, n_graphconv_layer=4,
                  n_linear_layer=1, linear_hidden_channels=[128], graphconv_dropout=0.012732466797412492,
                  linear_dropout=0.25, dataset="NNN_v2", architecture="GraphTransformer", concat=False)
    model = GNN(config).to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.eval()
    print(f"GNN device: {device}, torch threads: {torch.get_num_threads()}")

    t = time.perf_counter()
    sub = df[run]
    graphs = [row2graphdata(pd.Series({"RefSeq": s, "TargetStruct": t_, "dH": 0.0, "Tm": 0.0}),
                            sumstats_dict=sumstats, method="normalize")
              for s, t_ in zip(sub.sequence, sub.structure)]
    tm["graphs"] = time.perf_counter() - t
    t = time.perf_counter()
    pred = np.zeros((0, 2))
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=a.batch_size, shuffle=False):
            o = model(b.x.to(device), b.edge_index.to(device), b.edge_attr.to(device), b.batch.to(device))
            pred = np.concatenate([pred, o.cpu().numpy().reshape(-1, 2)])
    pred = unorm(pred, sumstats)
    tm["gnn"] = time.perf_counter() - t

    df["dH_pred_kcal_mol"] = np.nan
    df["Tm_pred_C"] = np.nan
    df.loc[run, "dH_pred_kcal_mol"] = pred[:, 0]
    df.loc[run, "Tm_pred_C"] = pred[:, 1]
    df["dG37_pred_kcal_mol"] = df.dH_pred_kcal_mol * (1 - 310.15 / (df.Tm_pred_C + 273.15))

    cols = ["file", "name", "sequence", "structure", "structure_source", "nupack_mfe_energy", "length",
            "dH_pred_kcal_mol", "Tm_pred_C", "dG37_pred_kcal_mol", "in_train_length_range", "status"]
    df = df[cols].round({"dH_pred_kcal_mol": 3, "Tm_pred_C": 3, "dG37_pred_kcal_mol": 3, "nupack_mfe_energy": 3})
    for f, g in df.groupby("file", sort=False):
        g.drop(columns="file").to_csv(outdir / f"{f}.gnn_pred.csv", index=False)
    df.to_csv(outdir / "all_predictions.csv", index=False)
    n_off = int((run & ~df.in_train_length_range).sum())
    tm["total"] = time.perf_counter() - T0
    n = max(int(run.sum()), 1)
    print("Timing (s): " + ", ".join(f"{k} {v:.2f}" for k, v in tm.items()) +
          f"  ->  {n / max(tm['total'], 1e-9):.0f} seq/s overall, {n / max(tm['gnn'], 1e-9):.0f} seq/s GNN only")
    print(f"Done: {int(run.sum())}/{len(df)} predicted -> {outdir}/ "
          f"(train length range {lmin}-{lmax} nt; {n_off} predicted sequences outside it, "
          f"{int((~run).sum())} skipped)")


if __name__ == "__main__":
    main()
