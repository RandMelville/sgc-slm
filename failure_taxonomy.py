"""Failure taxonomy of non-conformant outputs (reads archived results only).

No API and no Ollama needed. Reads results/conformity_<domain>.json, applies the same
de-dup rule as analysis.py (a successful attempt overrides a re-fetch that errored), keeps the
five core instruct models, and classifies every non-conformant call into one of five
categories by the validator message:

  invalid enum value         value outside the declared enum (split into accent/case variants
                             of a valid value versus out-of-vocabulary values)
  wrong type or shape        value present but of the wrong type (e.g. scalar where a list
                             is required), or wrong root type
  missing key                a required key is absent from the response; reported together
                             with how many of them are misspelled variants of the expected
                             key (e.g. "pergigos_reflexivas" for "perguntas_reflexivas")
  empty required string      a required string that is present but empty
  malformed JSON             response is not parseable JSON
  timeout or transport error read timeout (180 s) or connection failure

Also reports the qwen2.5:3b-instruct few-shot regression at education K3 (79% to 8%):
which enum value the model emitted under each strategy.

Usage:
    python3 failure_taxonomy.py            # writes results/failure_taxonomy.json
"""
import difflib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import contracts_clinical as clin
import contracts_education as edu

HERE = Path(__file__).resolve().parent
CORE = ["llama3.2:3b", "qwen2.5:3b-instruct", "gemma2:2b", "phi3:mini", "qwen3:4b-instruct"]
INPUTS = {"education": "results/conformity_education.json",
          "clinical": "results/conformity_clinical.json"}
ENUMS = {"education": set(edu.K3_NIVEIS) | set(edu.K3_TIPOS),
         "clinical": set(clin.K3_SISTEMAS) | set(clin.K3_URGENCIA)}
CATS = ["invalid enum value", "wrong type or shape", "missing key", "empty required string",
        "malformed JSON", "timeout or transport error"]
ENUM_FIELD = re.compile(r"(nivel_texto|nivel_urgencia|\.tipo|\.sistema)=(.*)$")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def load(path):
    best = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        k = (r["modelo"], r["condicao"], r["contrato"], r["cenario_id"], r["seed"])
        prev = best.get(k)
        if prev is None or ("erro" in prev and "erro" not in r):
            best[k] = r
    return [r for r in best.values() if r["modelo"] in CORE]


def _lookup(obj, path):
    """Walk a dotted path through nested dicts; returns (found, parent, last_key)."""
    parts = path.split(".")
    for key in parts[:-1]:
        if not isinstance(obj, dict) or key not in obj:
            return False, obj, parts[-1]
        obj = obj[key]
    return isinstance(obj, dict) and parts[-1] in obj and obj[parts[-1]] is not None, obj, parts[-1]


def classify(r):
    """Returns (category, misspelled) where misspelled is True for a near-miss key."""
    if "erro" in r:
        return "timeout or transport error", False
    m = r["motivo"]
    if m == "JSON inválido":
        return "malformed JSON", False
    if re.fullmatch(r"(pontos_fortes|achados)=str", m):
        return "empty required string", False  # a str that fails the non-emptiness check
    if re.search(r"(AUSENTE|=None$|vazio/ausente)", m):
        return "missing key", False
    if ENUM_FIELD.search(m):
        return "invalid enum value", False
    mm = re.fullmatch(r"([\w.]+)!=(?:list\[str\]|str)", m)  # e.g. perguntas_reflexivas!=list[str]
    if mm:
        try:
            obj = json.loads(r["resposta_ia"])
        except ValueError:
            return "malformed JSON", False
        found, parent, last = _lookup(obj, mm.group(1))
        if not found:
            close = isinstance(parent, dict) and difflib.get_close_matches(last, list(parent), n=1, cutoff=0.8)
            return "missing key", bool(close)
    return "wrong type or shape", False


def enum_kind(domain, motivo):
    val = ENUM_FIELD.search(motivo).group(2).strip("'\"")
    if strip_accents(val) in {strip_accents(x) for x in ENUMS[domain]}:
        return "accent or case variant"
    return "out of vocabulary"


def main():
    out = {}
    for dom, rel in INPUTS.items():
        rows = load(HERE / rel)
        calls, fails, kinds, oov = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
        typo = Counter()
        for r in rows:
            calls[r["condicao"]] += 1
            if "erro" in r or not r["conforme"]:
                cat, misspelled = classify(r)
                fails[r["condicao"]][cat] += 1
                typo[r["condicao"]] += int(misspelled)
                if cat == "invalid enum value":
                    kinds[r["condicao"]][enum_kind(dom, r["motivo"])] += 1
                    if enum_kind(dom, r["motivo"]) == "out of vocabulary":
                        oov[ENUM_FIELD.search(r["motivo"]).group(2).strip("'\"")] += 1
        out[dom] = {"calls_per_strategy": dict(calls),
                    "failures": {s: dict(c) for s, c in fails.items()},
                    "enum_failure_kinds": {s: dict(c) for s, c in kinds.items()},
                    "misspelled_key_failures": dict(typo),
                    "top_out_of_vocabulary": oov.most_common(8)}
        print(f"== {dom}: calls per strategy {dict(calls)}")
        print(f"{'category':28s} {'native':>7s} {'fewshot':>8s} {'grammar':>8s}")
        for cat in CATS:
            print(f"{cat:28s} {fails['native'][cat]:7d} {fails['fewshot'][cat]:8d} {fails['grammar'][cat]:8d}")
        print(f"   of the missing-key failures, misspelled near-miss keys: {dict(typo)}")
        print(f"   enum failures by kind: { {s: dict(c) for s, c in kinds.items()} }")
        print(f"   top out-of-vocabulary values: {oov.most_common(8)}")

    # qwen2.5:3b-instruct education K3: which nivel_texto value was emitted
    emitted = {}
    for con in ("native", "fewshot", "grammar"):
        c = Counter()
        for r in load(HERE / INPUTS["education"]):
            if r["modelo"] == "qwen2.5:3b-instruct" and r["contrato"] == "K3" \
                    and r["condicao"] == con and "erro" not in r:
                try:
                    c[json.loads(r["resposta_ia"]).get("nivel_texto")] += 1
                except ValueError:
                    c["<malformed>"] += 1
        emitted[con] = dict(c)
    out["qwen25_education_K3_nivel_texto"] = emitted
    print(f"\nqwen2.5:3b-instruct education K3, nivel_texto emitted: {emitted}")

    (HERE / "results" / "failure_taxonomy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
