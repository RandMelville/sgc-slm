"""Três contratos de saída estruturada, de complexidade crescente.

Objeto do experimento: comparar três caminhos para a conformidade estrutural
de um SLM — nativo (JSON-mode), few-shot (1 exemplo tipado) e grammar/schema-
constrained — medindo conformidade, qualidade de conteúdo e custo computacional.

Cada contrato traz:
  - system_prompt : a instrução (usada nas condições native e fewshot)
  - schema        : JSON Schema para o `format` do Ollama (condição grammar)
  - fewshot_user  : texto-demo do aluno (fora do conjunto de teste)
  - fewshot_gold  : resposta-ouro conforme o contrato (demo few-shot)
  - validate(p)   : (bool, motivo) — conformidade estrutural determinística

Reaproveita a instrução pedagógica e o construto de conformidade do benchmark
do doutorado (slm-writing-feedback-tutor-ptbr): validador de tipos herdado de
`diagnosticar()`/`divergent()`. O ponto do paper é que JSON-mode valida SINTAXE,
não TIPOS — daí o gradiente K1->K3 e o braço grammar.
"""

# Instrução pedagógica compartilhada (destilada do SYSTEM_PROMPT canônico do Bento).
BASE_INSTRUCAO = """Você é o Bento, um tutor de Linguística Textual (Koch) para alunos do 8º-9º ano da escola pública brasileira. Você opera após o aluno escrever um texto curto.

REGRAS:
1. NUNCA dê a resposta pronta nem reescreva o texto corrigido.
2. Use linguagem acolhedora e adequada à idade (13-14 anos).
3. Foque em coesão e coerência (Koch, 2020): repetição lexical, pronominalização ambígua, conectivos contraditórios, marcadores temporais.
4. Valorize variedades linguísticas e oralidade; não aja como policial gramatical.
5. Quando couber, faça uma ponte com o Pensamento Computacional (lógica, sequência, condição)."""

# Texto-demo (few-shot). NÃO pertence ao conjunto de teste — assim as 8 cenas de
# avaliação são julgadas igualmente sob as 3 condições.
DEMO_USER = ("Minha redação sobre as férias: Eu fui na praia. Eu joguei bola. "
             "Eu comi sorvete. Eu nadei no mar. Foi muito bom as férias.")


def _str_ok(x):
    return isinstance(x, str) and bool(x.strip())


def _lista_de_str(x):
    return isinstance(x, list) and len(x) > 0 and all(_str_ok(i) for i in x)


# ---------------------------------------------------------------------------
# K1 — contrato plano (str + lista de str). É exatamente a forma em que a
# família Llama 3.2 quebrou no benchmark (emitiu pontos_fortes como lista).
# ---------------------------------------------------------------------------
K1_FORMATO = """
FORMATO (JSON estrito, sem texto fora):
{
  "pontos_fortes": "um parágrafo de texto",
  "perguntas_reflexivas": ["pergunta 1", "pergunta 2"]
}"""

K1_SCHEMA = {
    "type": "object",
    "properties": {
        "pontos_fortes": {"type": "string"},
        "perguntas_reflexivas": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["pontos_fortes", "perguntas_reflexivas"],
}

K1_GOLD = ("{\n"
           '  "pontos_fortes": "Você contou as férias numa sequência bem clara, dá pra imaginar cada momento: a praia, a bola, o sorvete, o mar. Essa clareza é uma força do texto.",\n'
           '  "perguntas_reflexivas": ["Lê em voz alta o comecinho de cada frase: \'Eu... Eu... Eu...\'. Que som isso dá quando se repete tantas vezes seguidas?", "Se cada \'Eu fui\', \'Eu joguei\' fosse um comando repetido num programa, dava pra juntar algumas ações numa frase só, sem começar tudo igual?"]\n'
           "}")


def k1_validate(p):
    if not isinstance(p, dict):
        return False, f"raiz={type(p).__name__}"
    pf = p.get("pontos_fortes")
    if not _str_ok(pf):
        t = "AUSENTE" if pf is None else ("list" if isinstance(pf, list) else type(pf).__name__)
        return False, f"pontos_fortes={t}"
    if not _lista_de_str(p.get("perguntas_reflexivas")):
        return False, "perguntas_reflexivas!=list[str]"
    return True, "OK"


# ---------------------------------------------------------------------------
# K2 — contrato aninhado (objeto dentro de objeto).
# ---------------------------------------------------------------------------
K2_FORMATO = """
FORMATO (JSON estrito, sem texto fora):
{
  "feedback": {
    "ponto_forte": "um parágrafo de texto",
    "foco_koch": "o fenômeno de coesão/coerência priorizado"
  },
  "perguntas_reflexivas": ["pergunta 1", "pergunta 2"]
}"""

K2_SCHEMA = {
    "type": "object",
    "properties": {
        "feedback": {
            "type": "object",
            "properties": {
                "ponto_forte": {"type": "string"},
                "foco_koch": {"type": "string"},
            },
            "required": ["ponto_forte", "foco_koch"],
        },
        "perguntas_reflexivas": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["feedback", "perguntas_reflexivas"],
}

K2_GOLD = ("{\n"
           '  "feedback": {\n'
           '    "ponto_forte": "Você contou as férias numa sequência clara, dá pra imaginar cada momento.",\n'
           '    "foco_koch": "coesão referencial: repetição do pronome \'eu\' no início das frases"\n'
           "  },\n"
           '  "perguntas_reflexivas": ["Lê o comecinho de cada frase: \'Eu... Eu... Eu...\'. Que som isso dá?", "Dava pra juntar algumas ações numa frase só, sem repetir \'Eu\' toda vez?"]\n'
           "}")


def k2_validate(p):
    if not isinstance(p, dict):
        return False, f"raiz={type(p).__name__}"
    fb = p.get("feedback")
    if not isinstance(fb, dict):
        return False, f"feedback={'AUSENTE' if fb is None else type(fb).__name__}"
    if not _str_ok(fb.get("ponto_forte")):
        return False, "feedback.ponto_forte!=str"
    if not _str_ok(fb.get("foco_koch")):
        return False, "feedback.foco_koch!=str"
    if not _lista_de_str(p.get("perguntas_reflexivas")):
        return False, "perguntas_reflexivas!=list[str]"
    return True, "OK"


# ---------------------------------------------------------------------------
# K3 — contrato com enums + lista de objetos tipados.
# ---------------------------------------------------------------------------
K3_NIVEIS = ["inicial", "intermediario", "avancado"]
K3_TIPOS = ["coesao", "coerencia", "pc"]

K3_FORMATO = """
FORMATO (JSON estrito, sem texto fora):
{
  "nivel_texto": "inicial | intermediario | avancado",
  "pontos_fortes": "um parágrafo de texto",
  "perguntas": [
    {"pergunta": "texto da pergunta", "tipo": "coesao | coerencia | pc"}
  ]
}"""

K3_SCHEMA = {
    "type": "object",
    "properties": {
        "nivel_texto": {"type": "string", "enum": K3_NIVEIS},
        "pontos_fortes": {"type": "string"},
        "perguntas": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "pergunta": {"type": "string"},
                    "tipo": {"type": "string", "enum": K3_TIPOS},
                },
                "required": ["pergunta", "tipo"],
            },
        },
    },
    "required": ["nivel_texto", "pontos_fortes", "perguntas"],
}

K3_GOLD = ("{\n"
           '  "nivel_texto": "inicial",\n'
           '  "pontos_fortes": "Você contou as férias numa sequência clara, dá pra imaginar cada momento: praia, bola, sorvete, mar.",\n'
           '  "perguntas": [\n'
           '    {"pergunta": "Lê o comecinho de cada frase: \'Eu... Eu... Eu...\'. Que som isso dá quando se repete tantas vezes?", "tipo": "coesao"},\n'
           '    {"pergunta": "Se cada \'Eu fui\', \'Eu joguei\' fosse um comando repetido, dava pra juntar ações numa frase só?", "tipo": "pc"}\n'
           "  ]\n"
           "}")


def k3_validate(p):
    if not isinstance(p, dict):
        return False, f"raiz={type(p).__name__}"
    if p.get("nivel_texto") not in K3_NIVEIS:
        return False, f"nivel_texto={p.get('nivel_texto')!r}"
    if not _str_ok(p.get("pontos_fortes")):
        return False, "pontos_fortes!=str"
    perg = p.get("perguntas")
    if not isinstance(perg, list) or len(perg) == 0:
        return False, "perguntas vazio/ausente"
    for i, item in enumerate(perg):
        if not isinstance(item, dict):
            return False, f"perguntas[{i}]!=obj"
        if not _str_ok(item.get("pergunta")):
            return False, f"perguntas[{i}].pergunta!=str"
        if item.get("tipo") not in K3_TIPOS:
            return False, f"perguntas[{i}].tipo={item.get('tipo')!r}"
    return True, "OK"


CONTRATOS = {
    "K1": {
        "id": "K1", "descricao": "plano (str + lista de str)",
        "system_prompt": BASE_INSTRUCAO + "\n" + K1_FORMATO,
        "schema": K1_SCHEMA, "fewshot_user": DEMO_USER, "fewshot_gold": K1_GOLD,
        "validate": k1_validate,
    },
    "K2": {
        "id": "K2", "descricao": "aninhado (objeto dentro de objeto)",
        "system_prompt": BASE_INSTRUCAO + "\n" + K2_FORMATO,
        "schema": K2_SCHEMA, "fewshot_user": DEMO_USER, "fewshot_gold": K2_GOLD,
        "validate": k2_validate,
    },
    "K3": {
        "id": "K3", "descricao": "enums + lista de objetos tipados",
        "system_prompt": BASE_INSTRUCAO + "\n" + K3_FORMATO,
        "schema": K3_SCHEMA, "fewshot_user": DEMO_USER, "fewshot_gold": K3_GOLD,
        "validate": k3_validate,
    },
}
