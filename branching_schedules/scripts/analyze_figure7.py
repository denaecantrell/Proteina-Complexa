"""Unique successes vs compute for the Figure 7 subset."""

import argparse, glob, os, re, subprocess, sys, tempfile
import pandas as pd

REPO = os.path.expanduser("~/Proteina-Complexa")
FOLDSEEK = os.path.expanduser("~/miniconda3/envs/complexa/bin/foldseek")

# success criteria, from result_analysis/binder_analysis.py:309
IPAE_MAX, IPAE_SCALE = 7.0, 31.0
PLDDT_MIN = 0.9
SCRMSD_MAX = 1.5
TM = 0.5

# 15.6 s/sample on A100 (paper Tab. 2) vs 19.4 measured here
A100 = 15.6 / 19.4

MAX_POINTS = 10          # clustering is rerun on the whole set at each point
ALGOS = ["best-of-n", "beam-search", "fk-steering"]
COLORS = {"best-of-n": "#2ca02c", "beam-search": "#d62728", "fk-steering": "#7b4fa0"}


def target_chains(task):
    """Chains belonging to the target. Not always A: 27_HER2_AAV is C."""
    txt = open(f"{REPO}/configs/targets/targets_dict.yaml").read()
    blk = re.search(rf"^\s+{re.escape(task)}:\s*$(.*?)(?=^\s+\S+:\s*$)", txt, re.M | re.S)
    spec = re.search(r"target_input:\s*['\"]?([^'\"\n]+)", blk.group(1)) if blk else None
    return set(re.findall(r"([A-Za-z])\d", spec.group(1))) if spec else {"A"}


def n_clusters(pdbs, chains):
    """Foldseek clusters over binder chains only."""
    if not pdbs:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        d = f"{tmp}/pdb"
        os.makedirs(d)
        for i, p in enumerate(pdbs):
            if not os.path.exists(p):
                continue
            # the PDBs are complexes; clustering both chains counts the target too
            keep = [l for l in open(p)
                    if not l.startswith(("ATOM", "HETATM")) or l[21] not in chains]
            if any(l.startswith("ATOM") for l in keep):
                open(f"{d}/{i:06d}.pdb", "w").writelines(keep)
        if not os.listdir(d):
            return 0
        r = subprocess.run([FOLDSEEK, "easy-cluster", d, f"{tmp}/clu", f"{tmp}/t",
                            "--alignment-type", "1", "--tmscore-threshold", str(TM),
                            "--threads", "4"], capture_output=True)
        rep = f"{tmp}/clu_rep_seq.fasta"
        if r.returncode or not os.path.exists(rep):
            print(f"  foldseek rc={r.returncode}, falling back to raw count", file=sys.stderr)
            return len(os.listdir(d))
        return sum(l.startswith(">") for l in open(rep))


def curve(ledger, panel, use_scrmsd=False):
    led = pd.read_csv(ledger)
    if "seed" in led and led.seed.nunique() < len(led):
        print(f"  {panel}: {led.seed.nunique()} seeds for {len(led)} runs -- duplicates?",
              file=sys.stderr)
    out = []
    for (tgt, algo), g in led.groupby(["target", "algorithm"], sort=False):
        g = g.sort_values("repeat")
        chains = target_chains(tgt)
        reps = g["repeat"].tolist()
        step = max(1, len(reps) // MAX_POINTS)
        mark = set(reps[::step]) | {reps[-1]}
        hits, nstruct = [], 0
        for _, row in g.iterrows():
            f = glob.glob(f"{REPO}/inference/*{row.run_name}/rewards_*.csv")
            if not f:
                continue
            d = pd.read_csv(f[0])
            nstruct += len(d)
            ok = (d.af2folding_i_pae <= IPAE_MAX / IPAE_SCALE) & (d.af2folding_plddt_log >= PLDDT_MIN)
            if use_scrmsd and "binder_scrmsd" in d:
                ok &= d.binder_scrmsd < SCRMSD_MAX
            hits += d[ok].pdb_path.tolist()
            if row["repeat"] not in mark:
                continue
            h = row.cumulative_s / 3600
            out.append(dict(panel=panel, target=tgt, algorithm=algo, repeat=row["repeat"],
                            hours=h, a100_hours=h * A100, af2=nstruct,
                            passing=len(hits), unique=n_clusters(hits, chains)))
            print(f"  {panel:4s} {tgt:10s} {algo:12s} r{row['repeat']:<3d} "
                  f"{nstruct:6d} af2 {len(hits):5d} pass {out[-1]['unique']:4d} uniq")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--easy", default=f"{REPO}/fig7_easy_ledger.csv")
    ap.add_argument("--hard", default=f"{REPO}/fig7_hard_ledger.csv")
    ap.add_argument("--x", choices=["af2", "a100", "hours"], default="af2")
    ap.add_argument("--scrmsd", action="store_true")
    ap.add_argument("-o", default="figure7")
    a = ap.parse_args()

    rows = []
    for panel, led in [("easy", a.easy), ("hard", a.hard)]:
        if os.path.exists(led):
            rows += curve(led, panel, a.scrmsd)
    if not rows:
        sys.exit("no data")
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.o}.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xcol = {"af2": "af2", "a100": "a100_hours", "hours": "hours"}[a.x]
    xlab = {"af2": "AF2 reward evaluations",
            "a100": f"A100-equivalent GPU hours (x{A100:.2f})",
            "hours": "GPU hours (RTX 6000 Ada)"}[a.x]
    panels = [p for p in ("easy", "hard") if p in set(res.panel)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.8), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        sub = res[res.panel == panel]
        for algo in ALGOS:
            s = sub[sub.algorithm == algo].sort_values(xcol)
            if len(s):
                ax.plot(s[xcol], s.unique, "o-", label=algo, color=COLORS[algo])
        ax.set_title(f"{panel}: {', '.join(sorted(set(sub.target)))}", fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel("unique successes")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    crit = "ipAE<=7, pLDDT>=0.9" + (", scRMSD<1.5" if a.scrmsd else " (2 of 3 criteria)")
    fig.suptitle(f"Figure 7 subset -- {crit}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{a.o}.png", dpi=150)
    print(f"wrote {a.o}.csv and {a.o}.png")
