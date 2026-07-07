"""Harness do experimento de conformidade estrutural em SLMs.

Matriz fatorial: modelos x condicoes x contratos x cenarios x reps(seed).
Coleta contra Ollama local (http://localhost:11434). Salva de forma incremental
e RETOMÁVEL: se o arquivo de saída já existe, chamadas já feitas são puladas.

MULTI-DOMÍNIO: o mesmo harness roda em domínios diferentes via --dominio, trocando
apenas os contratos e os cenários. As estratégias, modelos, seeds e estatística são
idênticos entre domínios (é isso que torna o protocolo comparável e reutilizável).

Condições (3 caminhos para conformidade):
  native  -> format="json" (JSON-mode: valida sintaxe, não tipos) — a linha de base.
  fewshot -> format="json" + 1 exemplo tipado no histórico.
  grammar -> format=<JSON Schema tipado> (structured outputs do Ollama >= 0.5).

Determinismo: temperature fixa + options.seed por rep (42/43/44).

Uso típico:
    python3 harness.py --dry-run                 # educação (padrão), mostra a matriz
    python3 harness.py                           # roda educação (retomável)
    python3 harness.py --dominio medico          # roda a instância médica
    python3 harness.py --dominio medico --models qwen2.5:3b-instruct --contratos K1
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

# 5 modelos-núcleo (todos instruct/diretos, comparáveis). O qwen3:4b "thinking" NÃO
# entra no núcleo (é ~30x mais lento por raciocinar); ele é rodado à parte como uma
# sonda controlada de "reasoning tax" (ver README), já coletada em educação.
MODELOS = ["llama3.2:3b", "qwen2.5:3b-instruct", "gemma2:2b", "phi3:mini", "qwen3:4b-instruct"]
CONDICOES = ["native", "fewshot", "grammar"]
SEEDS = [42, 43, 44]  # 3 reps reprodutíveis
TEMPERATURE = 0.2

# Instâncias de domínio: (módulo de contratos, cenários, saída padrão).
# Educação mantém os caminhos originais para não quebrar rodadas em andamento.
DOMINIOS = {
    "educacao": {"contratos": "contratos", "cenarios": "cenarios.jsonl",
                 "out": "resultados/conformidade.json"},
    "medico": {"contratos": "contratos_medico", "cenarios": "cenarios_medico.jsonl",
               "out": "resultados/conformidade_medico.json"},
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


def diagnosticar(contrato, raw):
    try:
        p = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, "JSON inválido"
    return contrato["validate"](p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dominio", choices=list(DOMINIOS.keys()), default="educacao")
    ap.add_argument("--models", nargs="*", default=MODELOS)
    ap.add_argument("--condicoes", nargs="*", default=CONDICOES)
    ap.add_argument("--contratos", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--cenarios", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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
    print(f"Retomando: {len(feitos)} chamadas já concluídas, {total - len(feitos)} restantes.\n")

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
                            conforme, motivo = diagnosticar(contrato, r["resposta_ia"])
                            reg.update(r)
                            reg["conforme"] = conforme
                            reg["motivo"] = motivo
                            n_cel += 1
                            conf_cel += int(conforme)
                        except Exception as e:  # noqa: BLE001 — registra e segue
                            reg["erro"] = str(e)
                        resultados.append(reg)
                        feito_nesta_run += 1
                        # persistência incremental a cada chamada (crash-safe)
                        out_path.write_text(
                            json.dumps(resultados, ensure_ascii=False, indent=2),
                            encoding="utf-8")
                if n_cel:
                    print(f"{modelo:<22} {contrato_id} {condicao:<8} "
                          f"conforme {conf_cel}/{n_cel} ({100*conf_cel/n_cel:4.0f}%)")

    print(f"\nConcluído [{args.dominio}]. {feito_nesta_run} novas chamadas. "
          f"Total no arquivo: {len(resultados)}.")
    print(f"Saída: {out_path}")


if __name__ == "__main__":
    main()
