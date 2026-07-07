"""Juiz LLM (G-Eval pointwise) para BLINDAR o RQ2 — custo de qualidade.

Substitui os proxies léxicos (limite inferior por substring, saturados no
domínio médico) por um julgamento rubricado de um LLM-as-judge (Claude). A
avaliação é:
  - POINTWISE (nota 1–5 por resposta) → elimina viés de posição;
  - CEGA à estratégia que gerou a resposta (native/fewshot/grammar) → o juiz
    não sabe qual condição produziu o texto;
  - com CADEIA DE RACIOCÍNIO explícita antes da nota (G-Eval; Liu et al. 2023);
  - por um modelo DIFERENTE dos avaliados (SLMs ≤4B) → sem auto-preferência.

Fundamentação de validade (a citar no artigo): juízes fortes concordam ~80% com
anotadores humanos (Zheng et al. 2023, MT-Bench/Chatbot Arena) — nível de
concordância comparável ao inter-humano; G-Eval (Liu et al. 2023) formaliza a
rubrica + CoT pointwise; vieses conhecidos e mitigações em Zheng et al. 2023 e
Wang et al. 2023.

IMPORTANTE — o juiz é um INSTRUMENTO DE MEDIÇÃO OFFLINE, não parte do sistema
implantado. Os SLMs avaliados permanecem 100% locais; usar Claude como juiz é
análogo ao uso consolidado de GPT-4-as-judge em PLN e não altera a alegação de
"deployment local, sem API paga" (essa alega sobre o sistema estudado, não sobre
o aparato de avaliação).

Requer:  pip install anthropic scipy   e   export ANTHROPIC_API_KEY=...
Uso:
    python3 juiz.py --dominio educacao
    python3 juiz.py --dominio medico
Resumível e à prova de falha: grava juizo_<dominio>.json após cada julgamento e
pula os já feitos. Ao final, agrega grammar-vs-native (Mann-Whitney U) e grava
tabelas_juiz_<dominio>.json. Custo aproximado impresso ao iniciar.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent

MODELO_JUIZ = "claude-opus-4-8"  # juiz forte: é o regime em que a validade ~80% (Zheng) se aplica

DOMINIO_CFG = {
    "educacao": {
        "conf": "resultados/conformidade.json",
        "cenarios": "cenarios.jsonl",
        "rubrica": (
            "Você é um avaliador especialista em Linguística Textual (tradição de "
            "Ingedore Koch) e em feedback pedagógico de escrita para o ensino básico. "
            "Vai receber (a) o TEXTO DE UM ALUNO e (b) um FEEDBACK gerado por um "
            "assistente. Avalie SOMENTE a qualidade do feedback como retorno de "
            "escrita, com nota inteira de 1 a 5:\n"
            "5 = identifica corretamente problemas reais de coesão/coerência do TEXTO "
            "ESPECÍFICO (repetição, ambiguidade, referenciação, sequenciação, conexão), "
            "é pedagogicamente construtivo e acionável;\n"
            "3 = feedback pertinente porém genérico, parcial ou pouco acionável;\n"
            "1 = incorreto, alheio ao texto, ou sem valor pedagógico.\n"
            "Julgue o CONTEÚDO, não o formato. Primeiro raciocine em 1–2 frases, "
            "depois dê a nota."
        ),
    },
    "medico": {
        "conf": "resultados/conformidade_medico.json",
        "cenarios": "cenarios_medico.jsonl",
        "rubrica": (
            "Você é um enfermeiro(a) especialista em triagem clínica e segurança do "
            "paciente. Vai receber (a) uma VINHETA CLÍNICA e (b) uma resposta de um "
            "assistente de APOIO à triagem — cuja função é sinalizar achados e formular "
            "perguntas ao profissional, SEM diagnosticar nem prescrever. Avalie SOMENTE "
            "a qualidade do apoio, com nota inteira de 1 a 5:\n"
            "5 = sinaliza corretamente os achados salientes da vinheta, faz perguntas "
            "pertinentes ao profissional, permanece no escopo (não diagnostica/prescreve) "
            "e é clinicamente seguro;\n"
            "3 = parcialmente útil: perde achados relevantes OU é genérico, mas seguro;\n"
            "1 = perde achados críticos, extrapola o escopo (diagnóstico/prescrição) ou "
            "é clinicamente inseguro.\n"
            "Julgue o CONTEÚDO, não o formato. Primeiro raciocine em 1–2 frases, depois "
            "dê a nota."
        ),
    },
}

SCHEMA = {
    "type": "object",
    "properties": {
        "raciocinio": {"type": "string"},
        "nota": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["raciocinio", "nota"],
    "additionalProperties": False,
}


def carregar_cenarios(path):
    m = {}
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        m[d.get("id", i)] = d["user"]
    return m


def dedup_conformes(registros):
    """Um registro por chave, preferindo sucesso a erro; mantém só os CONFORMES."""
    best = {}
    for r in registros:
        k = (r["modelo"], r["condicao"], r["contrato"], r["cenario_id"], r["seed"])
        prev = best.get(k)
        if prev is None or ("erro" in prev and "erro" not in r):
            best[k] = r
    return [r for r in best.values() if "erro" not in r and r.get("conforme")]


def render_resposta(raw):
    """Renderiza o conteúdo de forma neutra (não revela a estratégia)."""
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw or ""


def julgar(client, rubrica, cenario_txt, resposta_txt):
    prompt = (
        f"TEXTO DO ALUNO / VINHETA CLÍNICA:\n{cenario_txt}\n\n"
        f"RESPOSTA DO ASSISTENTE (a avaliar):\n{resposta_txt}"
    )
    resp = client.messages.create(
        model=MODELO_JUIZ,
        max_tokens=1024,
        system=rubrica,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    obj = json.loads(text)
    return int(obj["nota"]), obj.get("raciocinio", "")


def agregar(juizos):
    """RQ2 blindado: por (modelo, contrato), compara grammar vs native (Mann-Whitney U)."""
    notas = defaultdict(list)  # (modelo, contrato, condicao) -> [notas]
    for j in juizos:
        notas[(j["modelo"], j["contrato"], j["condicao"])].append(j["nota"])

    modelos = sorted({j["modelo"] for j in juizos})
    contratos = sorted({j["contrato"] for j in juizos})
    linhas = []
    print("\n" + "=" * 88)
    print(f"RQ2 BLINDADO — nota do juiz (1–5) entre conformes; grammar vs native (Mann-Whitney)")
    print("=" * 88)
    print(f"{'modelo':<22} {'K':<3} {'nat n/méd':>12} {'gram n/méd':>12} {'U vs native':>16}")
    for m in modelos:
        for k in contratos:
            nat = notas.get((m, k, "native"), [])
            gram = notas.get((m, k, "grammar"), [])
            few = notas.get((m, k, "fewshot"), [])

            def stat(xs):
                return (len(xs), round(sum(xs) / len(xs), 2)) if xs else (0, None)

            n_nat, med_nat = stat(nat)
            n_gram, med_gram = stat(gram)
            n_few, med_few = stat(few)
            p = None
            if len(nat) >= 3 and len(gram) >= 3:
                # bilateral: detecta degradação OU melhora de qualidade sob grammar
                _, p = mannwhitneyu(gram, nat, alternative="two-sided")
            pstr = f"p={p:.3f}" if p is not None else "— (n baixo)"
            print(f"{m:<22} {k:<3} {n_nat:>4}/{str(med_nat):>6} "
                  f"{n_gram:>4}/{str(med_gram):>6} {pstr:>16}")
            linhas.append({
                "modelo": m, "contrato": k,
                "native_n": n_nat, "native_media": med_nat,
                "fewshot_n": n_few, "fewshot_media": med_few,
                "grammar_n": n_gram, "grammar_media": med_gram,
                "mannwhitney_grammar_vs_native_p": p,
            })
        print("-" * 88)

    # visão agregada (pooled) por condição
    pooled = defaultdict(list)
    for j in juizos:
        pooled[j["condicao"]].append(j["nota"])
    resumo = {c: {"n": len(v), "media": round(sum(v) / len(v), 3)} for c, v in pooled.items()}
    print(f"\nAgregado por condição (todas as células): {resumo}")
    return {"por_celula": linhas, "agregado_por_condicao": resumo, "modelo_juiz": MODELO_JUIZ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dominio", choices=list(DOMINIO_CFG.keys()), default="educacao")
    ap.add_argument("--limite", type=int, default=None, help="corta o nº de julgamentos (teste)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Defina ANTHROPIC_API_KEY (export ANTHROPIC_API_KEY=...).")
    import anthropic  # import tardio: só falha se a chave existir e for rodar

    cfg = DOMINIO_CFG[args.dominio]
    registros = json.loads((HERE / cfg["conf"]).read_text(encoding="utf-8"))
    conformes = dedup_conformes(registros)
    cenarios = carregar_cenarios(HERE / cfg["cenarios"])
    if args.limite:
        conformes = conformes[: args.limite]

    out = HERE / "resultados" / f"juizo_{args.dominio}.json"
    juizos = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    feitos = {(j["modelo"], j["condicao"], j["contrato"], j["cenario_id"], j["seed"]) for j in juizos}

    restantes = [r for r in conformes
                 if (r["modelo"], r["condicao"], r["contrato"], r["cenario_id"], r["seed"]) not in feitos]
    print(f"[{args.dominio}] conformes: {len(conformes)} | já julgados: {len(feitos)} | "
          f"a julgar: {len(restantes)}")
    print(f"Juiz: {MODELO_JUIZ}. Custo estimado ~ US$ {len(restantes) * 0.01:.2f} "
          f"(ordem de grandeza; ~1k tokens in + poucos out por julgamento).\n")

    client = anthropic.Anthropic()
    for i, r in enumerate(restantes, 1):
        cen = cenarios.get(r["cenario_id"], "")
        try:
            nota, racioc = julgar(client, cfg["rubrica"], cen, render_resposta(r.get("resposta_ia", "")))
        except Exception as e:  # noqa: BLE001 — registra e segue (resumível)
            print(f"  ! erro em {r['modelo']}/{r['condicao']}/{r['contrato']}/"
                  f"cen{r['cenario_id']}/s{r['seed']}: {e}")
            continue
        juizos.append({
            "dominio": args.dominio, "modelo": r["modelo"], "condicao": r["condicao"],
            "contrato": r["contrato"], "cenario_id": r["cenario_id"], "seed": r["seed"],
            "nota": nota, "raciocinio": racioc,
        })
        out.write_text(json.dumps(juizos, ensure_ascii=False, indent=2), encoding="utf-8")
        if i % 20 == 0 or i == len(restantes):
            print(f"  {i}/{len(restantes)} julgados…")

    print(f"\nJulgamentos salvos em {out} (total {len(juizos)}).")
    tabelas = agregar(juizos)
    tout = HERE / "resultados" / f"tabelas_juiz_{args.dominio}.json"
    tout.write_text(json.dumps(tabelas, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Tabelas RQ2 blindado salvas em {tout}")


if __name__ == "__main__":
    main()
