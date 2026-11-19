#!/usr/bin/env python3
"""Scenario-level (cluster) sensitivity analyses for the SGC study (no API, no Ollama).

Reads only the archived raw outputs (results/conformity_<domain>.json and
results/judgments_<domain>.json) and addresses the repeated-scenario structure
(8 scenarios x 3 seeds per cell), which the per-cell Fisher tests ignore.

  A. Conformity (RQ1). For every model x contract cell and comparison (few-shot or
     grammar versus native):
       - Fisher exact (as in analysis.py; treats the 24 calls as independent);
       - McNemar-type exact paired test on (scenario, seed)-matched calls;
       - scenario-level exact sign-flip permutation test on the 8 per-scenario
         differences in conformity rate (seeds collapsed inside the scenario).
     Also pooled differences (5 models together) per domain x contract with a
     scenario-cluster bootstrap 95% CI.
  B. Quality equivalence (RQ2). The paired grammar-minus-native judge gap among jointly
     conformant calls, aggregated at three levels (call, model x contract x scenario,
     scenario), with t-TOST and a scenario-cluster bootstrap; plus descriptive judge
     scores for the calls where grammar conforms and native does not.
  C. Latency (RQ3). Means and dispersion over returned responses, and per attempted call
     with the unresolved timeout/transport-error calls included at explicit assumed
     durations (the 180 s read timeout is only a lower bound).

Conventions follow analysis.py: duplicates are resolved in favour of a successful
attempt; a call that never succeeded counts as non-conformant; the qwen3:4b reasoning
probe is excluded (5-model instruct core).

Usage:  python3 sensitivity_r3.py            # prints everything, writes results/sensitivity_r3.json
"""
import itertools
import json
import os
import numpy as np
from collections import defaultdict
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DOMAINS = {"education": "educacao", "clinical": "medico"}
EXCLUDE = {"qwen3:4b"}
CONDS = ["native", "fewshot", "grammar"]
SCEN = list(range(1, 9))
SEEDS = [42, 43, 44]
TIMEOUT_MS = 180_000
BOOT_N = 10000
MARGIN = 0.25
ALPHA_BONF = 0.05 / 60
RNG = np.random.RandomState(0)
OUT = {}


# ----------------------------------------------------------------------------- loading
def load_conf(slug):
    raw = json.load(open(os.path.join(RES, f"conformity_{slug}.json")))
    best = {}
    order = {}
    for i, r in enumerate(raw):
        if r["modelo"] in EXCLUDE:
            continue
        k = (r["modelo"], r["condicao"], r["contrato"], int(r["cenario_id"]), int(r["seed"]))
        prev = best.get(k)
        if prev is None or ("erro" in prev and "erro" not in r):
            best[k] = r
            order[k] = i
    return best, order


def conf_vec(best):
    """(model, contract, cond) -> {(scenario, seed): 0/1}; failed call = 0."""
    v = defaultdict(dict)
    for (m, c, k, s, sd), r in best.items():
        v[(m, k, c)][(s, sd)] = 0 if "erro" in r else int(bool(r.get("conforme")))
    return v


# ------------------------------------------------------------------------------- tests
def fisher_p(a, b):
    ca, na, cb, nb = sum(a), len(a), sum(b), len(b)
    return stats.fisher_exact([[ca, na - ca], [cb, nb - cb]])[1]


def mcnemar_p(x, y):
    """Exact two-sided test on discordant (scenario, seed) pairs; x = comparison, y = native."""
    keys = sorted(set(x) & set(y))
    b = sum(1 for k in keys if x[k] == 1 and y[k] == 0)
    c = sum(1 for k in keys if x[k] == 0 and y[k] == 1)
    if b + c == 0:
        return 1.0, b, c
    return stats.binomtest(min(b, c), b + c, 0.5).pvalue, b, c


def signflip_p(d):
    """Exact sign-flip permutation test on scenario-level differences d (|sum| statistic)."""
    d = np.asarray([v for v in d if abs(v) > 1e-12], float)
    if len(d) == 0:
        return 1.0, 0
    obs = abs(d.sum())
    hits = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        if abs(np.dot(signs, d)) >= obs - 1e-12:
            hits += 1
    return hits / 2 ** len(d), len(d)


def scen_rate(x, s):
    return np.mean([x[(s, sd)] for sd in SEEDS if (s, sd) in x]) if any((s, sd) in x for sd in SEEDS) else np.nan


# ----------------------------------------------------------------------------------- A
def part_a():
    print("=" * 100)
    print("A. CONFORMITY: per-cell Fisher versus scenario-aware tests")
    print("=" * 100)
    rows = []
    pooled = []
    for dom, slug_key in DOMAINS.items():
        best, _ = load_conf(dom)
        v = conf_vec(best)
        models = sorted({m for (m, k, c) in v})
        contracts = sorted({k for (m, k, c) in v})
        for m in models:
            for k in contracts:
                nat = v[(m, k, "native")]
                for cond in ("fewshot", "grammar"):
                    x = v[(m, k, cond)]
                    ka = sorted(nat)
                    kx = sorted(x)
                    pf = fisher_p([nat[q] for q in ka], [x[q] for q in kx])
                    pm, b, c = mcnemar_p(x, nat)
                    d = [scen_rate(x, s) - scen_rate(nat, s) for s in SCEN]
                    d = [t for t in d if not np.isnan(t)]
                    ps, nz = signflip_p(d)
                    rows.append({"domain": dom, "model": m, "contract": k, "comparison": f"{cond}_vs_native",
                                 "native_pct": round(100 * np.mean(list(nat.values())), 1),
                                 "cond_pct": round(100 * np.mean(list(x.values())), 1),
                                 "fisher_p": pf, "mcnemar_p": pm, "disc_cond_only": b, "disc_native_only": c,
                                 "scenario_signflip_p": ps, "scenarios_with_diff": nz,
                                 "min_attainable_signflip_p": 2 / 2 ** nz if nz else 1.0})
        # pooled, scenario-cluster bootstrap
        for k in contracts:
            for cond in ("fewshot", "grammar"):
                def rate(cd, scen_ids):
                    vals = []
                    for m in models:
                        x = v[(m, k, cd)]
                        for s in scen_ids:
                            for sd in SEEDS:
                                if (s, sd) in x:
                                    vals.append(x[(s, sd)])
                    return np.mean(vals)
                diff = rate(cond, SCEN) - rate("native", SCEN)
                bs = []
                for _ in range(BOOT_N):
                    ids = RNG.choice(SCEN, len(SCEN), replace=True)
                    bs.append(rate(cond, ids) - rate("native", ids))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                pooled.append({"domain": dom, "contract": k, "comparison": f"{cond}_vs_native",
                               "diff_pp": round(float(100 * diff), 1), "ci95_pp": [round(float(100 * lo), 1), round(float(100 * hi), 1)]})
    n_fisher = sum(r["fisher_p"] < 0.05 for r in rows)
    bonf = [r for r in rows if r["fisher_p"] < ALPHA_BONF]
    print(f"tests: {len(rows)} | Fisher p<0.05: {n_fisher} | Fisher survives Bonferroni (p<{ALPHA_BONF:.5f}): {len(bonf)}")
    print(f"scenario sign-flip p<0.05: {sum(r['scenario_signflip_p'] < 0.05 for r in rows)} | "
          f"McNemar p<0.05: {sum(r['mcnemar_p'] < 0.05 for r in rows)}")
    print("\nFisher-Bonferroni survivors, with the scenario-aware tests (floor = smallest attainable sign-flip p):")
    print(f"{'dom':<9}{'model':<20}{'K':<3}{'cmp':<18}{'nat%':>6}{'cond%':>7}{'Fisher':>10}{'McNemar':>10}{'sgnflip':>9}{'floor':>8}")
    for r in sorted(bonf, key=lambda r: r["fisher_p"]):
        print(f"{r['domain']:<9}{r['model']:<20}{r['contract']:<3}{r['comparison']:<18}{r['native_pct']:>6}{r['cond_pct']:>7}"
              f"{r['fisher_p']:>10.1e}{r['mcnemar_p']:>10.1e}{r['scenario_signflip_p']:>9.4f}{r['min_attainable_signflip_p']:>8.4f}")
    sig = [r for r in rows if r["fisher_p"] < 0.05]
    lost = [r for r in sig if r["scenario_signflip_p"] >= 0.05]
    print(f"\nFisher p<0.05 cells that are NOT significant at scenario level (sign-flip p>=0.05): {len(lost)} of {len(sig)}")
    for r in lost:
        print(f"   {r['domain']} {r['model']} {r['contract']} {r['comparison']}: Fisher {r['fisher_p']:.1e}, "
              f"sign-flip {r['scenario_signflip_p']:.4f} (scenarios with a difference: {r['scenarios_with_diff']})")
    print("\nPooled over the 5 models, scenario-cluster bootstrap 95% CI of the conformity difference (percentage points):")
    for p in pooled:
        print(f"   {p['domain']:<9}{p['contract']:<3}{p['comparison']:<18}{p['diff_pp']:>7.1f}  CI {p['ci95_pp']}")
    OUT["A_conformity"] = {"cells": rows, "pooled": pooled}


# ----------------------------------------------------------------------------------- B
def tost_p(x, margin):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    if se == 0:
        return 0.0 if abs(x.mean()) < margin else 1.0
    df = len(x) - 1
    return max(stats.t.sf((x.mean() + margin) / se, df), stats.t.cdf((x.mean() - margin) / se, df))


def min_margin(x, alpha=0.05):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(1 - alpha, len(x) - 1)
    return max(abs(x.mean() - t * se), abs(x.mean() + t * se))


def part_b():
    print("\n" + "=" * 100)
    print("B. QUALITY EQUIVALENCE with scenario-level dependence")
    print("=" * 100)
    out = {}
    for dom in DOMAINS:
        jud = [r for r in json.load(open(os.path.join(RES, f"judgments_{dom}.json"))) if r["modelo"] not in EXCLUDE]
        best, _ = load_conf(dom)
        v = conf_vec(best)
        nat, gram = {}, {}
        for r in jud:
            key = (r["modelo"], r["contrato"], int(r["cenario_id"]), int(r["seed"]))
            {"native": nat, "grammar": gram}.get(r["condicao"], {})[key] = r["nota"]
        common = sorted(set(nat) & set(gram))
        diffs = {k: gram[k] - nat[k] for k in common}
        # level 0: call
        d0 = np.array(list(diffs.values()), float)
        # level 1: model x contract x scenario (seeds collapsed)
        g1 = defaultdict(list)
        for (m, k, s, sd), d in diffs.items():
            g1[(m, k, s)].append(d)
        d1 = np.array([np.mean(x) for x in g1.values()])
        # level 2: scenario
        g2 = defaultdict(list)
        for (m, k, s, sd), d in diffs.items():
            g2[s].append(d)
        d2 = np.array([np.mean(g2[s]) for s in sorted(g2)])
        # scenario-cluster bootstrap of the pooled call-level mean gap
        sc = sorted(g2)
        bs = []
        for _ in range(BOOT_N):
            ids = RNG.choice(sc, len(sc), replace=True)
            bs.append(np.mean(np.concatenate([g2[s] for s in ids])))
        lo95, hi95 = np.percentile(bs, [2.5, 97.5])
        lo90, hi90 = np.percentile(bs, [5, 95])
        print(f"\n--- {dom.upper()} ---  jointly conformant & judged calls: {len(common)}  "
              f"(native judged {len(nat)}, grammar judged {len(gram)})")
        res = {"n_joint": len(common), "levels": {}}
        for name, x in (("call", d0), ("model_contract_scenario", d1), ("scenario", d2)):
            row = {"n_units": len(x), "mean_gap": round(float(x.mean()), 3),
                   "tost_p_0.25": round(float(tost_p(x, MARGIN)), 4),
                   "min_margin_alpha05": round(float(min_margin(x)), 3),
                   "tost_p_sweep": {str(m): round(float(tost_p(x, m)), 4) for m in (0.10, 0.15, 0.20, 0.25, 0.30)}}
            res["levels"][name] = row
            print(f"   {name:<26} n={len(x):<4} gap={row['mean_gap']:+.3f}  TOST(±0.25) p={row['tost_p_0.25']:.4f}  "
                  f"smallest margin={row['min_margin_alpha05']:.3f}  sweep {row['tost_p_sweep']}")
        res["scenario_cluster_bootstrap"] = {"mean_gap": round(float(np.mean(np.concatenate([g2[s] for s in sc]))), 3),
                                              "ci95": [round(float(lo95), 3), round(float(hi95), 3)],
                                              "ci90": [round(float(lo90), 3), round(float(hi90), 3)],
                                              "ci90_within_margin_0.25": bool(lo90 > -MARGIN and hi90 < MARGIN)}
        print(f"   scenario-cluster bootstrap: gap {res['scenario_cluster_bootstrap']['mean_gap']:+.3f} "
              f"95% CI {res['scenario_cluster_bootstrap']['ci95']}, 90% CI {res['scenario_cluster_bootstrap']['ci90']} "
              f"-> inside ±{MARGIN}: {res['scenario_cluster_bootstrap']['ci90_within_margin_0.25']}")
        print("   per-scenario mean gap (grammar - native):", {s: round(float(np.mean(g2[s])), 2) for s in sc})

        # descriptive: grammar conformant (judged) where matched native is NOT conformant
        g_only, joint_g = [], []
        for key, nota in gram.items():
            m, k, s, sd = key
            nat_ok = v[(m, k, "native")].get((s, sd), 0)
            (joint_g if nat_ok else g_only).append((k, nota))
        n_only = [(k, nat[key]) for key in nat for k in [key[1]] if not v[(key[0], key[1], "grammar")].get((key[2], key[3]), 0)]
        desc = {"grammar_ok_native_not": {"n": len(g_only), "mean": round(float(np.mean([n for _, n in g_only])), 3) if g_only else None,
                                           "by_contract": {k: {"n": sum(1 for kk, _ in g_only if kk == k),
                                                                "mean": round(float(np.mean([n for kk, n in g_only if kk == k])), 3)}
                                                           for k in sorted({kk for kk, _ in g_only})}},
                "grammar_ok_native_ok": {"n": len(joint_g), "mean": round(float(np.mean([n for _, n in joint_g])), 3) if joint_g else None,
                                          "by_contract": {k: {"n": sum(1 for kk, _ in joint_g if kk == k),
                                                              "mean": round(float(np.mean([n for kk, n in joint_g if kk == k])), 3)}
                                                          for k in sorted({kk for kk, _ in joint_g})}},
                "native_ok_grammar_not": {"n": len(n_only), "mean": round(float(np.mean([n for _, n in n_only])), 3) if n_only else None}}
        res["descriptive_discordant"] = desc
        byk = {}
        for k in ("K1", "K2", "K3"):
            gk = defaultdict(list)
            for (m, kk, s, sd), d in diffs.items():
                if kk == k:
                    gk[s].append(d)
            if not gk:
                continue
            sck = sorted(gk)
            bsk = [np.mean(np.concatenate([gk[s] for s in RNG.choice(sck, len(sck), replace=True)])) for _ in range(BOOT_N)]
            allk = np.concatenate([gk[s] for s in sck])
            byk[k] = {"n_pairs": int(len(allk)), "mean_gap": round(float(allk.mean()), 3),
                      "ci95": [round(float(np.percentile(bsk, 2.5)), 3), round(float(np.percentile(bsk, 97.5)), 3)]}
        res["paired_gap_by_contract"] = byk
        print("   paired gap by contract (scenario-cluster bootstrap 95% CI):", byk)
        print(f"   descriptive judge scores (grammar, 1-5): grammar conforms & native does not: n={desc['grammar_ok_native_not']['n']}, "
              f"mean={desc['grammar_ok_native_not']['mean']}, by contract {desc['grammar_ok_native_not']['by_contract']}")
        print(f"                                            both conform: n={desc['grammar_ok_native_ok']['n']}, mean={desc['grammar_ok_native_ok']['mean']}, by contract {desc['grammar_ok_native_ok']['by_contract']}")
        print(f"                              native conforms & grammar does not: n={desc['native_ok_grammar_not']['n']}, mean={desc['native_ok_grammar_not']['mean']}")
        out[dom] = res
    OUT["B_equivalence"] = out


# ----------------------------------------------------------------------------------- C
def ci_mean(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def part_c():
    print("\n" + "=" * 100)
    print("C. LATENCY: returned responses versus per attempted call (failed calls included)")
    print("=" * 100)
    out = {"unresolved_failed_calls": [], "cells": [], "first_call_per_model": []}
    for dom in DOMAINS:
        best, order = load_conf(dom)
        failed = [k for k, r in best.items() if "erro" in r]
        for k in failed:
            out["unresolved_failed_calls"].append({"domain": dom, "model": k[0], "condition": k[1], "contract": k[2],
                                                   "scenario": k[3], "seed": k[4]})
        print(f"\n{dom}: unresolved failed calls (never succeeded): {len(failed)}")
        for k in sorted(failed):
            print("   ", k)
        cells = defaultdict(list)
        for k, r in best.items():
            if "erro" not in r and r.get("latencia_ms") is not None:
                cells[(k[0], k[2], k[1])].append(r["latencia_ms"] / 1000)
        for (m, k, c), n_failed in defaultdict(int, {(f[0], f[2], f[1]): 0 for f in failed}).items():
            pass
        nfail = defaultdict(int)
        for f in failed:
            nfail[(f[0], f[2], f[1])] += 1
        # per-cell summaries
        models = sorted({m for (m, _, _) in cells})
        contracts = sorted({k for (_, k, _) in cells})
        print(f"{'model':<20}{'K':<3}{'cond':<8}{'n_ret':>6}{'fail':>5}{'mean_s':>8}{'sd':>7}{'95%CI':>16}{'mean+fail@180':>15}{'@360':>8}{'@900':>8}")
        for m in models:
            for k in contracts:
                for c in CONDS:
                    x = cells.get((m, k, c), [])
                    if not x:
                        continue
                    nf = nfail.get((m, k, c), 0)
                    lo, hi = ci_mean(x)
                    imp = lambda t: (sum(x) + nf * t) / (len(x) + nf)
                    row = {"domain": dom, "model": m, "contract": k, "condition": c, "n_returned": len(x), "n_failed": nf,
                           "mean_s": round(float(np.mean(x)), 2), "sd_s": round(float(np.std(x, ddof=1)), 2),
                           "ci95_s": [round(float(lo), 2), round(float(hi), 2)],
                           "mean_with_failed_180": round(imp(180), 2), "mean_with_failed_360": round(imp(360), 2),
                           "mean_with_failed_900": round(imp(900), 2)}
                    out["cells"].append(row)
                    flag = "  <-- affected" if nf else ""
                    print(f"{m:<20}{k:<3}{c:<8}{len(x):>6}{nf:>5}{row['mean_s']:>8.2f}{row['sd_s']:>7.2f}"
                          f"{str([row['ci95_s'][0], row['ci95_s'][1]]):>16}{row['mean_with_failed_180']:>15.2f}"
                          f"{row['mean_with_failed_360']:>8.2f}{row['mean_with_failed_900']:>8.2f}{flag}")
        # first call per model (warm-up / residency evidence): file order, first returned call
        first = {}
        for k, i in sorted(order.items(), key=lambda t: t[1]):
            r = best[k]
            if "erro" in r or r["modelo"] in first:
                continue
            first[r["modelo"]] = (k, r["latencia_ms"] / 1000)
        for m, (k, lat) in sorted(first.items()):
            same = [x for (mm, kk, cc), xs in cells.items() if mm == m and kk == k[2] and cc == k[1] for x in xs]
            out["first_call_per_model"].append({"domain": dom, "model": m, "first_call_key": list(k), "first_call_s": round(lat, 2),
                                                "median_same_cell_s": round(float(np.median(same)), 2)})
        print(f"\n{dom}: first returned call per model (file order) versus median of its cell")
        for row in [r for r in out["first_call_per_model"] if r["domain"] == dom]:
            print(f"   {row['model']:<20} first={row['first_call_s']:>7.2f}s  cell median={row['median_same_cell_s']:>6.2f}s  ({row['first_call_key'][1:]})")

    # parity summary: grammar / native ratio of mean latency, across the 5 core models, by domain and contract
    print("\nGrammar/native latency ratio (pooled call means), returned responses only vs failed calls included at 180 s:")
    for dom in DOMAINS:
        rows = [r for r in out["cells"] if r["domain"] == dom]
        for k in sorted({r["contract"] for r in rows}):
            def pooled(cond, key):
                cs = [r for r in rows if r["contract"] == k and r["condition"] == cond]
                if key == "ret":
                    num = sum(r["mean_s"] * r["n_returned"] for r in cs); den = sum(r["n_returned"] for r in cs)
                else:
                    num = sum(r["mean_with_failed_180"] * (r["n_returned"] + r["n_failed"]) for r in cs)
                    den = sum(r["n_returned"] + r["n_failed"] for r in cs)
                return num / den
            print(f"   {dom:<10}{k}: native {pooled('native','ret'):.2f}s, grammar {pooled('grammar','ret'):.2f}s "
                  f"(ratio {pooled('grammar','ret')/pooled('native','ret'):.2f})  |  with failed@180: native {pooled('native','imp'):.2f}s, "
                  f"grammar {pooled('grammar','imp'):.2f}s (ratio {pooled('grammar','imp')/pooled('native','imp'):.2f})")
    # tail latency of returned calls by strategy, and the call that failed once and then succeeded
    out["tail_latency"] = []
    out["retried_then_succeeded"] = []
    print("\nTail latency of returned calls by strategy (s):")
    for dom in DOMAINS:
        raw = json.load(open(os.path.join(RES, f"conformity_{dom}.json")))
        best, _ = load_conf(dom)
        errs = {}
        for r in raw:
            if r["modelo"] in EXCLUDE:
                continue
            k = (r["modelo"], r["condicao"], r["contrato"], int(r["cenario_id"]), int(r["seed"]))
            if "erro" in r:
                errs[k] = errs.get(k, 0) + 1
        for k, n in errs.items():
            if "erro" not in best[k]:
                out["retried_then_succeeded"].append({"domain": dom, "model": k[0], "condition": k[1], "contract": k[2],
                                                      "scenario": k[3], "seed": k[4], "failed_attempts_before_success": n,
                                                      "latency_of_successful_attempt_s": round(best[k]["latencia_ms"] / 1000, 2)})
        for c in CONDS:
            xs = np.array([r["latencia_ms"] / 1000 for k, r in best.items() if k[1] == c and "erro" not in r])
            row = {"domain": dom, "condition": c, "n": int(len(xs)), "median_s": round(float(np.median(xs)), 2),
                   "p95_s": round(float(np.percentile(xs, 95)), 2), "p99_s": round(float(np.percentile(xs, 99)), 2),
                   "max_s": round(float(xs.max()), 2)}
            out["tail_latency"].append(row)
            print(f"   {dom:<10}{c:<8} n={row['n']:<4} median {row['median_s']:>6} p95 {row['p95_s']:>6} p99 {row['p99_s']:>6} max {row['max_s']:>7}")
    print("Retried then succeeded:", out["retried_then_succeeded"])

    # pooled over all contracts and the five core models (the figures quoted in the text)
    out["pooled_all_contracts"] = []
    print("\nPooled over all contracts and the 5 core models (mean seconds per call):")
    for dom in DOMAINS:
        best, order = load_conf(dom)
        first = {}
        for k, i in sorted(order.items(), key=lambda t: t[1]):
            if "erro" not in best[k] and k[0] not in first:
                first[k[0]] = k
        firstset = set(first.values())

        def pm(cond, fail=None, skipfirst=False, drop=None):
            xs, nf = [], 0
            for k, r in best.items():
                if k[1] != cond or (drop and k[0] == drop):
                    continue
                if "erro" in r:
                    nf += 1
                    continue
                if skipfirst and k in firstset:
                    continue
                xs.append(r["latencia_ms"] / 1000)
            return (sum(xs) + (fail or 0) * nf) / (len(xs) + (nf if fail else 0)), len(xs), nf

        row = {"domain": dom}
        for label, kw in (("returned", {}), ("failed_at_180s", {"fail": 180}), ("failed_at_360s", {"fail": 360}),
                          ("first_call_per_model_dropped", {"skipfirst": True}), ("excluding_phi3_returned", {"drop": "phi3:mini"})):
            n, g = pm("native", **kw), pm("grammar", **kw)
            row[label] = {"native_s": round(n[0], 2), "grammar_s": round(g[0], 2), "ratio": round(g[0] / n[0], 2)}
            print(f"   {dom:<10}{label:<30} native {n[0]:.2f}  grammar {g[0]:.2f}  ratio {g[0]/n[0]:.2f}")
        out["pooled_all_contracts"].append(row)
    OUT["C_latency"] = out


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    path = os.path.join(RES, "sensitivity_r3.json")
    json.dump(OUT, open(path, "w"), ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved {path}")
