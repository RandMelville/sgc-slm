"""Three structured-output contracts of increasing complexity (EDUCATION domain).

The experiment compares three paths to structural conformity for an SLM: native
(JSON-mode), few-shot (one typed exemplar), and grammar/schema-constrained, while
measuring conformity, content quality, and compute cost.

Each contract provides:
  - system_prompt : the instruction (used in the native and few-shot conditions)
  - schema        : JSON Schema for Ollama's `format` (grammar condition)
  - fewshot_user  : demo student text (outside the test set)
  - fewshot_gold  : gold response conforming to the contract (few-shot demo)
  - validate(p)   : (bool, reason) -- deterministic structural conformity

JSON-mode validates SYNTAX, not TYPES; hence the K1->K3 gradient and the grammar arm.
Prompts, scenarios, and gold responses are kept in Brazilian Portuguese: they are the
experimental stimuli for a Brazilian basic-education domain and must not be translated.
"""

# Shared pedagogical instruction (system prompt; the tutor persona is named "Bento").
BASE_INSTRUCAO = """Você é o Bento, um tutor de Linguística Textual (Koch) para alunos do 8º-9º ano da escola pública brasileira. Você opera após o aluno escrever um texto curto.

REGRAS:
1. NUNCA dê a resposta pronta nem reescreva o texto corrigido.
2. Use linguagem acolhedora e adequada à idade (13-14 anos).
3. Foque em coesão e coerência (Koch, 2020): repetição lexical, pronominalização ambígua, conectivos contraditórios, marcadores temporais.
4. Valorize variedades linguísticas e oralidade; não aja como policial gramatical.
5. Quando couber, faça uma ponte com o Pensamento Computacional (lógica, sequência, condição)."""

# Few-shot demo text. NOT part of the test set, so the 8 evaluation scenarios are
# judged equally under all 3 conditions.
DEMO_USER = ("Minha redação sobre as férias: Eu fui na praia. Eu joguei bola. "
             "Eu comi sorvete. Eu nadei no mar. Foi muito bom as férias.")


def _str_ok(x):
    return isinstance(x, str) and bool(x.strip())


def _lista_de_str(x):
    return isinstance(x, list) and len(x) > 0 and all(_str_ok(i) for i in x)


# ---------------------------------------------------------------------------
# K1 -- flat contract (str + list of str). This is exactly the shape where the
# Llama 3.2 family broke in a prior benchmark (emitting pontos_fortes as a list).
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
# K2 -- nested contract (object within object).
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
# K3 -- enums + list of typed objects.
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
        "id": "K1", "descricao": "flat (str + list of str)",
        "system_prompt": BASE_INSTRUCAO + "\n" + K1_FORMATO,
        "schema": K1_SCHEMA, "fewshot_user": DEMO_USER, "fewshot_gold": K1_GOLD,
        "validate": k1_validate,
    },
    "K2": {
        "id": "K2", "descricao": "nested (object within object)",
        "system_prompt": BASE_INSTRUCAO + "\n" + K2_FORMATO,
        "schema": K2_SCHEMA, "fewshot_user": DEMO_USER, "fewshot_gold": K2_GOLD,
        "validate": k2_validate,
    },
    "K3": {
        "id": "K3", "descricao": "enums + list of typed objects",
        "system_prompt": BASE_INSTRUCAO + "\n" + K3_FORMATO,
        "schema": K3_SCHEMA, "fewshot_user": DEMO_USER, "fewshot_gold": K3_GOLD,
        "validate": k3_validate,
    },
}
