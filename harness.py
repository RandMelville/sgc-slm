"""Experiment harness for structural conformity in SLMs.

Factorial matrix: models x conditions x contracts x scenarios x reps(seed).
Collects against local Ollama (http://localhost:11434). Saves incrementally and is
RESUMABLE: if the output file already exists, calls already made are skipped.

MULTI-DOMAIN: the same harness runs on different domains via --domain, swapping only
the contracts and scenarios. Strategies, models, seeds, and statistics are identical
across domains (this is what makes the protocol comparable and reusable).

Conditions (3 paths to conformity):
  native  -> format="json" (JSON-mode: validates syntax, not types) -- the baseline.
  fewshot -> format="json" + 1 typed exemplar in the history.
  grammar -> format=<typed JSON Schema> (Ollama structured outputs, >= 0.5).

Determinism: fixed temperature + options.seed per rep (42/43/44).

Typical usage:
    python3 harness.py --dry-run                 # education (default), shows the matrix
    python3 harness.py                           # runs education (resumable)
    python3 harness.py --domain clinical         # runs the clinical instance
    python3 harness.py --domain clinical --models qwen2.5:3b-instruct --contracts K1

Domain keys are stored internally in Portuguese ("educacao", "medico") because they
are recorded inside the archived result files; the command line accepts the English
names and the original Portuguese ones interchangeably.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OLLAMA = "http://localhost:11434/api/chat"

# 5 core models (all instruct/direct, comparable). The qwen3:4b "thinking" variant is
# NOT part of the core (it is ~30x slower because it reasons); it is run separately as a
# controlled "reasoning tax" probe (see README), already collected for education.
MODELOS = ["llama3.2:3b", "qwen2.5:3b-instruct", "gemma2:2b", "phi3:mini", "qwen3:4b-instruct"]
CONDICOES = ["native", "fewshot", "grammar"]
SEEDS = [42, 43, 44]  # 3 reproducible reps
TEMPERATURE = 0.2

# Domain instances: (contracts module, scenarios, default output path).
DOMINIOS = {
    "educacao": {"contratos": "contracts_education",
                 "cenarios": "scenarios_education.jsonl",
                 "out": "results/conformity_education.json"},
    "medico": {"contratos": "contracts_clinical",
               "cenarios": "scenarios_clinical.jsonl",
               "out": "results/conformity_clinical.json"},
}
# English command-line names mapped onto the canonical internal keys.
DOMAIN_ALIASES = {
    "education": "educacao", "educacao": "educacao",
    "clinical": "medico", "medico": "medico",
}


def carregar_cenarios(path):
    cenarios = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        cenarios.append({"id": d.get("id", i), "user": d["user"]})
    return cenarios


def montar_messages(contrato, condicao, user_text):
    msgs = [{"role": "system", "content": contrato["system_prompt"]}]
    if condicao == "fewshot":
        msgs.append({"role": "user", "content": contrato["fewshot_user"]})
        msgs.append({"role": "assistant", "content": contrato["fewshot_gold"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def chamar(modelo, messages, condicao, contrato, seed, timeout):
    fmt = contrato["schema"] if condicao == "grammar" else "json"
    payload = {
        "model": modelo,
        "messages": messages,
        "stream": False,
        "format": fmt,
        "options": {"temperature": TEMPERATURE, "seed": seed},
    }
    t0 = time.time()
    r = requests.post(OLLAMA, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return {
        "latencia_ms": int((time.time() - t0) * 1000),
        "resposta_ia": data["message"]["content"],
        "tokens_in": data.get("prompt_eval_count"),
        "tokens_out": data.get("eval_count"),
    }


def chave(r):
    return (r["modelo"], r["condicao"], r["contrato"], r["cenario_id"], r["seed"])


def diagnose(contrato, raw):
    try:
        p = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, "JSON inválido"
    return contrato["validate"](p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", "--dominio", dest="dominio",
                    choices=list(DOMAIN_ALIASES.keys()), default="education")
    ap.add_argument("--models", nargs="*", default=MODELOS)
    ap.add_argument("--conditions", "--condicoes", dest="condicoes",
                    nargs="*", default=CONDICOES)
    ap.add_argument("--contracts", "--contratos", dest="contratos",
                    nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--scenarios", "--cenarios", dest="cenarios", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.dominio = DOMAIN_ALIASES[args.dominio]

    cfg = DOMINIOS[args.dominio]
    CONTRATOS = importlib.import_module(cfg["contratos"]).CONTRATOS
    cenarios_path = args.cenarios or str(HERE / cfg["cenarios"])
    out_path = Path(args.out or str(HERE / cfg["out"]))
    contratos_ids = args.contratos or list(CONTRATOS.keys())

    cenarios = carregar_cenarios(cenarios_path)
    total = (len(args.models) * len(args.condicoes) * len(contratos_ids)
             * len(cenarios) * len(args.seeds))
    print(f"[dominio={args.dominio}] Matriz: {len(args.models)} modelos x "
          f"{len(args.condicoes)} condicoes x {len(contratos_ids)} contratos x "
          f"{len(cenarios)} cenarios x {len(args.seeds)} reps = {total} chamadas")
    if args.dry_run:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    resultados = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    feitos = {chave(r) for r in resultados if "erro" not in r}
    print(f"Resuming: {len(feitos)} calls already completed, {total - len(feitos)} remaining.\n")

    feito_nesta_run = 0
    for modelo in args.models:
        for contrato_id in contratos_ids:
            contrato = CONTRATOS[contrato_id]
            for condicao in args.condicoes:
                conf_cel = n_cel = 0
                for c in cenarios:
                    for seed in args.seeds:
                        k = (modelo, condicao, contrato_id, c["id"], seed)
                        if k in feitos:
                            continue
                        reg = {"dominio": args.dominio, "modelo": modelo, "condicao": condicao,
                               "contrato": contrato_id, "cenario_id": c["id"], "seed": seed}
                        try:
                            r = chamar(modelo, montar_messages(contrato, condicao, c["user"]),
                                       condicao, contrato, seed, args.timeout)
                            conforme, motivo = diagnose(contrato, r["resposta_ia"])
                            reg.update(r)
                            reg["conforme"] = conforme
                            reg["motivo"] = motivo
                            n_cel += 1
                            conf_cel += int(conforme)
                        except Exception as e:  # noqa: BLE001 -- record and continue
                            reg["erro"] = str(e)
                        resultados.append(reg)
                        feito_nesta_run += 1
                        # incremental persistence on each call (crash-safe)
                        out_path.write_text(
                            json.dumps(resultados, ensure_ascii=False, indent=2),
                            encoding="utf-8")
                if n_cel:
                    print(f"{modelo:<22} {contrato_id} {condicao:<8} "
                          f"conformant {conf_cel}/{n_cel} ({100*conf_cel/n_cel:4.0f}%)")

    print(f"\nDone [{args.dominio}]. {feito_nesta_run} new calls. "
          f"Total in file: {len(resultados)}.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
