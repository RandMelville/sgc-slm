"""Analysis of the SLM structural-conformity experiment (multi-domain).

Reads resultados/conformidade[_domain].json and answers the three questions:
  RQ1 -- conformity by model x condition x contract (rate + Wilson CI);
         Fisher's exact native-vs-grammar and native-vs-few-shot.
  RQ2 -- quality cost: among CONFORMANT responses, does the grammar condition
         degrade content (domain-lexicon adherence) vs native?
  RQ3 -- compute cost: latency and output tokens per condition.

Per-domain quality lexicon: education = Koch cohesion/coherence taxonomy;
clinical = objective clinical terminology. Both are LOWER-BOUND substring proxies
(they measure presence, not adequacy); human (kappa) validation is future work.

Fisher's exact and Wilson CI follow a prior SLM writing-feedback benchmark. Timeouts
count as non-conformant (conservative). RQ2 guard: the per-cell conformant n is
reported, and quality comparisons must not be read where native is near zero.

Usage:
    python3 analise.py                    # education (default)
    python3 analise.py --dominio medico
"""
import argparse
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent

# --- Per-domain quality proxies (normalized stems, substring matching).
LEXICOS = {
    "educacao": {  # Koch cohesion/coherence taxonomy
        "coes", "conect", "conjun", "pronom", "referenc", "retomad", "sequenci",
        "ambigu", "repet", "sinon", "elipse", "elipt", "argument", "marcador",
        "temporal", "justapos",
    },
    "medico": {  # objective clinical terminology (findings/signs/systems)
        "sintom", "sinal", "satura", "press", "frequenc", "febr", "dor", "dispne",
        "edem", "glic", "cardio", "respirat", "neuro", "metaboli", "hipertens",
        "taquic", "cefale", "confus", "urgenc", "triag", "comorbid", "auscult",
        "irradi", "sudore", "poliur",
    },
}
DOMINIO_INPUT = {
    "educacao": "resultados/conformidade.json",
    "medico": "resultados/conformidade_medico.json",
}


def normalize(text):
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def texto_gerado(raw):
    """Extract all free text from the response, tolerating any contract/malformation."""
    try:
        p = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw or ""
    partes = []

    def walk(x):
        if isinstance(x, str):
            partes.append(x)
        elif isinstance(x, list):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(p)
    return " ".join(partes) if partes else (raw or "")


def adere(raw, stems):
    norm = normalize(texto_gerado(raw))
    return any(stem in norm for stem in stems)


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, center - half) * 100, 1), round(min(1.0, center + half) * 100, 1))


def fisher(a_c, a_n, b_c, b_n):
    _, p = fisher_exact([[a_c, a_n - a_c], [b_c, b_n - b_c]], alternative="two-sided")
    return p


def fmt_p(p):
    if p is None:
        return "—"
    if p < 0.001:
        return "p<0.001 ***"
    if p < 0.01:
        return f"p={p:.3f} **"
    if p < 0.05:
        return f"p={p:.3f} *"
    return f"p={p:.3f} n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dominio", choices=list(LEXICOS.keys()), default="educacao")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stems = LEXICOS[args.dominio]
    inp = Path(args.inp or str(HERE / DOMINIO_INPUT[args.dominio]))
    out = Path(args.out or str(HERE / "resultados" / f"tabelas_{args.dominio}.json"))

    registros = json.loads(inp.read_text(encoding="utf-8"))
    # De-dup by key: a successful attempt overrides a re-fetch that errored,
    # so each (model, condition, contract, scenario, seed) counts exactly once.
    best = {}
    for r in registros:
        k = (r["modelo"], r["condicao"], r["contrato"], r["cenario_id"], r["seed"])
        prev = best.get(k)
        if prev is None or ("erro" in prev and "erro" not in r):
            best[k] = r
    registros = list(best.values())
    modelos = sorted({r["modelo"] for r in registros})
    contratos = sorted({r["contrato"] for r in registros})
    condicoes = ["native", "fewshot", "grammar"]

    agg = defaultdict(lambda: {"c": 0, "n": 0, "lat": [], "tok": [], "q_ad": 0, "q_n": 0})
    for r in registros:
        key = (r["modelo"], r["contrato"], r["condicao"])
        a = agg[key]
        a["n"] += 1
        if "erro" in r:  # timeout/error counts as non-conformant (denominator)
            continue
        conforme = bool(r.get("conforme"))
        a["c"] += int(conforme)
        if r.get("latencia_ms") is not None:
            a["lat"].append(r["latencia_ms"])
        if r.get("tokens_out") is not None:
            a["tok"].append(r["tokens_out"])
        if conforme:  # RQ2: quality only among conformant responses
            a["q_n"] += 1
            a["q_ad"] += int(adere(r.get("resposta_ia", ""), stems))

    def med(xs):
        return round(sum(xs) / len(xs)) if xs else None

    print(f"\n### DOMAIN: {args.dominio}  (input: {inp.name})")

    # ---------------- RQ1 ----------------
    print("=" * 92)
    print("RQ1 -- Structural conformity by model x contract x condition")
    print("=" * 92)
    print(f"{'model':<22} {'K':<3} {'condition':<8} {'conf.':>8} {'rate':>6} "
          f"{'Wilson 95%':>14} {'Fisher vs native':>18}")
    tabela_rq1 = []
    for m in modelos:
        for k in contratos:
            base = agg[(m, k, "native")]
            for cond in condicoes:
                a = agg[(m, k, cond)]
                if a["n"] == 0:
                    continue
                lo, hi = wilson_ci(a["c"], a["n"])
                taxa = round(100 * a["c"] / a["n"], 1)
                if cond == "native":
                    fstr, p = "(reference)", None
                else:
                    p = fisher(a["c"], a["n"], base["c"], base["n"]) if base["n"] else None
                    fstr = fmt_p(p)
                print(f"{m:<22} {k:<3} {cond:<8} {a['c']:>4}/{a['n']:<3} {taxa:>5.0f}% "
                      f"[{lo:>4.0f};{hi:>4.0f}] {fstr:>18}")
                tabela_rq1.append({"modelo": m, "contrato": k, "condicao": cond,
                                   "conformes": a["c"], "n": a["n"], "taxa_pct": taxa,
                                   "wilson_lo": lo, "wilson_hi": hi,
                                   "fisher_vs_native": fstr, "p": p})
        print("-" * 92)

    # ---------------- RQ2 ----------------
    print("\n" + "=" * 92)
    print(f"RQ2 -- Quality cost: lexicon adherence ({args.dominio}) AMONG conformant")
    print("(guard: ignore comparisons where native's conformant n is ~0)")
    print("=" * 92)
    print(f"{'model':<22} {'K':<3} {'condition':<8} {'adherence(conf.)':>18} {'rate':>6}"
          f"{'Fisher vs native':>18}")
    tabela_rq2 = []
    for m in modelos:
        for k in contratos:
            base = agg[(m, k, "native")]
            for cond in condicoes:
                a = agg[(m, k, cond)]
                if a["q_n"] == 0:
                    continue
                taxa = round(100 * a["q_ad"] / a["q_n"], 1)
                if cond == "native" or base["q_n"] == 0:
                    fstr, p = "(reference)" if cond == "native" else "—", None
                else:
                    p = fisher(a["q_ad"], a["q_n"], base["q_ad"], base["q_n"])
                    fstr = fmt_p(p)
                print(f"{m:<22} {k:<3} {cond:<8} {a['q_ad']:>7}/{a['q_n']:<8} {taxa:>5.0f}% {fstr:>18}")
                tabela_rq2.append({"modelo": m, "contrato": k, "condicao": cond,
                                   "aderentes": a["q_ad"], "n_conformes": a["q_n"],
                                   "taxa_pct": taxa, "fisher_vs_native": fstr, "p": p})
        print("-" * 92)

    # ---------------- RQ3 ----------------
    print("\n" + "=" * 92)
    print("RQ3 -- Compute cost: mean latency (ms) and output tokens per condition")
    print("=" * 92)
    print(f"{'model':<22} {'K':<3} {'condition':<8} {'lat.mean(ms)':>14} {'tokens.out':>12}")
    tabela_rq3 = []
    for m in modelos:
        for k in contratos:
            for cond in condicoes:
                a = agg[(m, k, cond)]
                if not a["lat"]:
                    continue
                lm, tm = med(a["lat"]), med(a["tok"])
                print(f"{m:<22} {k:<3} {cond:<8} {lm:>14} {tm:>12}")
                tabela_rq3.append({"modelo": m, "contrato": k, "condicao": cond,
                                   "latencia_media_ms": lm, "tokens_out_medio": tm})
        print("-" * 92)

    out.write_text(json.dumps({"dominio": args.dominio, "rq1": tabela_rq1,
                               "rq2": tabela_rq2, "rq3": tabela_rq3},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nTables saved to {out}")


if __name__ == "__main__":
    main()
