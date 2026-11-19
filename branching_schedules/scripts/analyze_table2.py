"""Collect the Table 2 row for each target that has been through analyze."""

import ast, glob, os, sys
import pandas as pd

REPO = os.path.expanduser("~/Proteina-Complexa")
OUT = os.path.expanduser("~/Proteina-Complexa/branching_schedules/results")

# paper Tab. 2, Complexa "Self" column, mean over 19 targets
PAPER_UNIQUE, PAPER_TIME = 9.10, 15.6


def row(d):
    """One target's numbers from its evaluation_results directory."""
    tgt = os.path.basename(d).replace("binder_eval_table2_", "")
    f = f"{d}/overall_binder_performance_evaluate_aggregated.csv"
    if not os.path.exists(f):
        return None
    x = pd.read_csv(f).iloc[0]

    n = int(x["_res_self_original_samples_overall"])
    rate = float(x["_res_self_pass_rate_success_overall"])
    # stored as a literal "(ratio, n_clusters, n_items)"
    _, uniq, succ = ast.literal_eval(x["_res_diversity_foldseek_binder_successful_self"])

    # generate-only runs write total_time; generate+evaluate runs write
    # generation_time AND a total_time that includes evaluation. The paper's
    # Time column is generation only, so prefer generation_time.
    t = glob.glob(f"{REPO}/inference/*{tgt}_table2_{tgt}/timing_*.csv")
    per_sample = None
    if t:
        td = pd.read_csv(t[0])
        col = "generation_time" if "generation_time" in td else "total_time"
        if col in td and "nsamples" in td:
            per_sample = float(td[col].iloc[0]) / float(td["nsamples"].iloc[0])

    return dict(target=tgt, samples=n, pass_rate=round(rate, 4),
                successes=succ, n_unique=uniq,
                s_per_sample=round(per_sample, 1) if per_sample else None)


if __name__ == "__main__":
    dirs = sorted(glob.glob(f"{REPO}/evaluation_results/binder_eval_table2_*"))
    rows = [r for r in (row(d) for d in dirs) if r]
    if not rows:
        sys.exit("no analyzed Table 2 runs found")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/table2.csv", index=False)

    w = [f"Table 2: base generation without optimization ({len(df)} targets)",
         "",
         f"{'target':<14}{'samples':>9}{'pass rate':>11}{'successes':>11}{'unique':>8}{'s/sample':>10}"]
    for _, r in df.iterrows():
        w.append(f"{r.target:<14}{r.samples:>9}{r.pass_rate:>10.1%}"
                 f"{r.successes:>11}{r["n_unique"]:>8}"
                 f"{(r.s_per_sample if r.s_per_sample else float('nan')):>10.1f}")
    w += ["",
          f"{'mean':<14}{'':>9}{'':>11}{'':>11}{df["n_unique"].mean():>8.1f}"
          f"{df.s_per_sample.mean():>10.1f}",
          f"{'paper (19)':<14}{'':>9}{'':>11}{'':>11}{PAPER_UNIQUE:>8.2f}{PAPER_TIME:>10.1f}",
          "",
          "unique = Foldseek clusters over designs passing all three criteria.",
          "Per-sample time is generation only and is the one column comparable",
          "across hardware; the paper's figure is an A100.",
          ]
    if len(df) < 19:
        w.append(f"Only {len(df)} of 19 targets, so the mean is not comparable to 9.10.")

    txt = "\n".join(w)
    open(f"{OUT}/table2.txt", "w").write(txt + "\n")
    print(txt)
    print(f"\nwrote {OUT}/table2.csv and {OUT}/table2.txt")
