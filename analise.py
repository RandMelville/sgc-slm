"""Análise do experimento de conformidade estrutural em SLMs (multi-domínio).

Lê resultados/conformidade[_dominio].json e responde às três perguntas:
  RQ1 — conformidade por modelo x condicao x contrato (taxa + IC de Wilson);
        Fisher exato native-vs-grammar e native-vs-fewshot.
  RQ2 — custo de qualidade: entre as respostas CONFORMES, a condição grammar
        degrada o conteúdo (aderência ao léxico do domínio) vs native?
  RQ3 — custo computacional: latência e tokens de saída por condição.

Léxico de qualidade por domínio: educação = taxonomia de Koch (coesão/coerência);
médico = terminologia clínica objetiva. Ambos são PROXIES de limite inferior por
substring (não medem adequação, só presença) — validação humana (κ) é trabalho futuro.

Estatística (Fisher exato, IC de Wilson) reaproveitada do benchmark do doutorado.
Timeouts contam como não-conformes (conservador). Guard de RQ2: o n-conforme por
célula é reportado, e comparações de qualidade não devem ser lidas onde native≈0.

Uso:
    python3 analise.py                    # educação (padrão)
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

# --- Proxies de qualidade por domínio (stems normalizados, casamento por substring).
LEXICOS = {
    "educacao": {  # taxonomia de coesão/coerência de Koch (fiel ao scorer do doutorado)
        "coes", "conect", "conjun", "pronom", "referenc", "retomad", "sequenci",
        "ambigu", "repet", "sinon", "elipse", "elipt", "argument", "marcador",
        "temporal", "justapos",
    },
    "medico": {  # terminologia clínica objetiva (achados/sinais/sistemas)
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
    """Extrai todo o texto livre da resposta, tolerando qualquer contrato/malformação."""
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
    # De-dup por chave: uma tentativa bem-sucedida sobrepõe uma repesca que deu erro,
    # para que cada (modelo,condição,contrato,cenário,seed) conte exatamente uma vez.
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
        if "erro" in r:  # timeout/erro conta como não-conforme (denominador)
            continue
        conforme = bool(r.get("conforme"))
        a["c"] += int(conforme)
        if r.get("latencia_ms") is not None:
            a["lat"].append(r["latencia_ms"])
        if r.get("tokens_out") is not None:
            a["tok"].append(r["tokens_out"])
        if conforme:  # RQ2: qualidade só entre conformes
            a["q_n"] += 1
            a["q_ad"] += int(adere(r.get("resposta_ia", ""), stems))

    def med(xs):
        return round(sum(xs) / len(xs)) if xs else None

    print(f"\n### DOMÍNIO: {args.dominio}  (entrada: {inp.name})")

    # ---------------- RQ1 ----------------
    print("=" * 92)
    print("RQ1 — Conformidade estrutural por modelo x contrato x condição")
    print("=" * 92)
    print(f"{'modelo':<22} {'K':<3} {'condição':<8} {'conf.':>8} {'taxa':>6} "
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
                    fstr, p = "(referência)", None
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
    print(f"RQ2 — Custo de qualidade: aderência ao léxico ({args.dominio}) ENTRE conformes")
    print("(guard: ignore comparações onde o n-conforme do native é ~0)")
    print("=" * 92)
    print(f"{'modelo':<22} {'K':<3} {'condição':<8} {'aderência(conf.)':>18} {'taxa':>6} "
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
                    fstr, p = "(referência)" if cond == "native" else "—", None
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
    print("RQ3 — Custo computacional: latência média (ms) e tokens de saída por condição")
    print("=" * 92)
    print(f"{'modelo':<22} {'K':<3} {'condição':<8} {'lat.média(ms)':>14} {'tokens.out':>12}")
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
    print(f"\nTabelas salvas em {out}")


if __name__ == "__main__":
    main()
