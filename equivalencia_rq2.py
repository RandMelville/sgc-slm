#!/usr/bin/env python3
"""RQ2 equivalence / robustness check (no API, no Ollama).

Reads the blind-judge raw outputs (resultados/juizo_<dominio>.json) and reports,
for grammar vs. native quality among conformant responses:

  (a) the naive aggregate gap (as in Table 2), and
  (b) a scenario-matched PAIRED gap restricted to (model x contract x scenario x
      seed) cells where BOTH strategies conform, with a 95% bootstrap CI,
      Wilcoxon signed-rank, and a TOST equivalence test against a +-0.25 margin.

The paired result guards the null against the survivorship confound (native's
conformant set is easy-biased on collapsing cells). The reasoning-tax probe model
(qwen3:4b, education-only) is excluded to match the paper's 5-model core.

Usage:  python3 equivalencia_rq2.py
"""
import json, os, numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(__file__), "resultados")
EXCLUDE = {"qwen3:4b"}          # reasoning-tax probe, outside the 5-model instruct core
MARGIN = 0.25                    # smallest quality effect of interest (1-5 scale)
BOOT_N = 10000
RNG = np.random.RandomState(0)   # fixed for reproducibility


def load(dom):
    path = os.path.join(RESULTS, f"juizo_{dom}.json")
    return [r for r in json.load(open(path)) if r["modelo"] not in EXCLUDE]


def cell_key(r):
    return (r["modelo"], r["contrato"], r["cenario_id"], r["seed"])


def boot_ci(x):
    x = np.asarray(x, float)
    idx = RNG.randint(0, len(x), (BOOT_N, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def tost(diff, margin):
    """Two one-sided t-tests; returns max p (H1: |mean| < margin)."""
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    df = len(diff) - 1
    p_low = stats.t.sf((diff.mean() + margin) / se, df)   # H0: mean <= -margin
    p_high = stats.t.cdf((diff.mean() - margin) / se, df)  # H0: mean >= +margin
    return max(p_low, p_high)


def main():
    for dom in ["educacao", "medico"]:
        recs = load(dom)
        by_cond = {}
        for r in recs:
            by_cond.setdefault(r["condicao"], {})[cell_key(r)] = r["nota"]
        nat, gram = by_cond["native"], by_cond["grammar"]

        nat_all = np.array(list(nat.values()), float)
        gram_all = np.array(list(gram.values()), float)

        common = sorted(set(nat) & set(gram))
        dn = np.array([nat[k] for k in common], float)
        dg = np.array([gram[k] for k in common], float)
        diff = dg - dn

        ci = boot_ci(diff)
        _, wp = stats.wilcoxon(dg, dn)
        tost_p = tost(diff, MARGIN)

        print(f"\n=== {dom.upper()} ===")
        print(f"aggregate  : native n={len(nat_all)} mean={nat_all.mean():.3f} | "
              f"grammar n={len(gram_all)} mean={gram_all.mean():.3f} | "
              f"gap={gram_all.mean()-nat_all.mean():+.3f}")
        print(f"paired     : n={len(common)} | native={dn.mean():.3f} grammar={dg.mean():.3f} "
              f"| gap={diff.mean():+.3f} | 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"Wilcoxon p={wp:.3f} | TOST(+-{MARGIN}) p={tost_p:.4f} -> "
              f"{'EQUIVALENT' if tost_p < 0.05 else 'inconclusive'}")


if __name__ == "__main__":
    main()
