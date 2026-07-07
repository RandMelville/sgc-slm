"""Clinical instance of the contracts (second domain of the study).

Same structure as `contratos.py` (education): three contracts of increasing complexity
(K1 flat -> K2 nested -> K3 enums + typed list), each with a system prompt, JSON Schema,
few-shot exemplar, and deterministic per-contract validator.

Task: clinical-triage SUPPORT from a short, SYNTHETIC clinical note. The model flags
findings and raises questions for the professional; it does not diagnose or prescribe.
This "support, not decision-maker" framing is deliberate: it instantiates verifiable,
human-in-the-loop AI in a distinct high-risk domain.

IMPORTANT: all scenarios are fictitious/synthetic; no real personal data is processed
(LGPD art. 5, I). This is not medical advice; the object of study is structural output
conformity, not real triage. Prompts and scenarios are kept in Brazilian Portuguese as
the experimental stimuli.
"""

BASE_INSTRUCAO = """Você é um assistente de APOIO à triagem clínica. A partir de uma nota clínica curta, você sinaliza achados objetivos e levanta perguntas para o profissional de saúde decidir. Você opera como apoio, nunca como decisor.

REGRAS:
1. NUNCA dê diagnóstico definitivo nem prescreva conduta ou medicação.
2. Sinalize apenas achados objetivos presentes na nota (sinais vitais, sintomas, tempo de evolução).
3. Levante perguntas que ajudem o profissional a decidir; não substitua o julgamento clínico.
4. Não invente dados que não estão na nota.
5. Linguagem técnica, concisa e objetiva."""

DEMO_USER = ("Paciente masculino, 40 anos, tosse seca há 5 dias, febre de 37,8°C, "
             "sem dispneia, saturação de 97% em ar ambiente.")


def _str_ok(x):
    return isinstance(x, str) and bool(x.strip())


def _lista_de_str(x):
    return isinstance(x, list) and len(x) > 0 and all(_str_ok(i) for i in x)


# --- K1: flat (str + list of str) -----------------------------------------
K1_FORMATO = """
FORMATO (JSON estrito, sem texto fora):
{
  "achados": "um parágrafo com os achados objetivos",
  "perguntas_ao_profissional": ["pergunta 1", "pergunta 2"]
}"""

K1_SCHEMA = {
    "type": "object",
    "properties": {
        "achados": {"type": "string"},
        "perguntas_ao_profissional": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["achados", "perguntas_ao_profissional"],
}

K1_GOLD = ("{\n"
           '  "achados": "Tosse seca de 5 dias com febre baixa (37,8°C), sem dispneia e com saturação preservada (97%). Quadro respiratório leve, sem sinais de gravidade evidentes na nota.",\n'
           '  "perguntas_ao_profissional": ["Há histórico de contato com sintomáticos respiratórios ou comorbidades relevantes?", "A ausculta pulmonar apresenta alterações que justifiquem exame de imagem?"]\n'
           "}")


def k1_validate(p):
    if not isinstance(p, dict):
        return False, f"raiz={type(p).__name__}"
    ac = p.get("achados")
    if not _str_ok(ac):
        t = "AUSENTE" if ac is None else ("list" if isinstance(ac, list) else type(ac).__name__)
        return False, f"achados={t}"
    if not _lista_de_str(p.get("perguntas_ao_profissional")):
        return False, "perguntas_ao_profissional!=list[str]"
    return True, "OK"


# --- K2: nested ------------------------------------------------------------
K2_FORMATO = """
FORMATO (JSON estrito, sem texto fora):
{
  "avaliacao": {
    "achado_principal": "o achado mais relevante",
    "categoria": "o sistema/categoria do achado"
  },
  "perguntas_ao_profissional": ["pergunta 1", "pergunta 2"]
}"""

K2_SCHEMA = {
    "type": "object",
    "properties": {
        "avaliacao": {
            "type": "object",
            "properties": {
                "achado_principal": {"type": "string"},
                "categoria": {"type": "string"},
            },
            "required": ["achado_principal", "categoria"],
        },
        "perguntas_ao_profissional": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["avaliacao", "perguntas_ao_profissional"],
}

K2_GOLD = ("{\n"
           '  "avaliacao": {\n'
           '    "achado_principal": "Quadro respiratório leve: tosse seca e febre baixa, sem sinais de gravidade.",\n'
           '    "categoria": "respiratorio"\n'
           "  },\n"
           '  "perguntas_ao_profissional": ["Há comorbidades ou imunossupressão?", "A ausculta indica necessidade de exame de imagem?"]\n'
           "}")


def k2_validate(p):
    if not isinstance(p, dict):
        return False, f"raiz={type(p).__name__}"
    av = p.get("avaliacao")
    if not isinstance(av, dict):
        return False, f"avaliacao={'AUSENTE' if av is None else type(av).__name__}"
    if not _str_ok(av.get("achado_principal")):
        return False, "avaliacao.achado_principal!=str"
    if not _str_ok(av.get("categoria")):
        return False, "avaliacao.categoria!=str"
    if not _lista_de_str(p.get("perguntas_ao_profissional")):
        return False, "perguntas_ao_profissional!=list[str]"
    return True, "OK"


# --- K3: enums + list of typed objects ------------------------------------
K3_URGENCIA = ["baixo", "medio", "alto"]
K3_SISTEMAS = ["cardio", "respiratorio", "neuro", "metabolico", "outro"]

K3_FORMATO = """
FORMATO (JSON estrito, sem texto fora):
{
  "nivel_urgencia": "baixo | medio | alto",
  "achados": "um parágrafo com os achados objetivos",
  "sinais": [
    {"sinal": "descrição do sinal", "sistema": "cardio | respiratorio | neuro | metabolico | outro"}
  ]
}"""

K3_SCHEMA = {
    "type": "object",
    "properties": {
        "nivel_urgencia": {"type": "string", "enum": K3_URGENCIA},
        "achados": {"type": "string"},
        "sinais": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "sinal": {"type": "string"},
                    "sistema": {"type": "string", "enum": K3_SISTEMAS},
                },
                "required": ["sinal", "sistema"],
            },
        },
    },
    "required": ["nivel_urgencia", "achados", "sinais"],
}

K3_GOLD = ("{\n"
           '  "nivel_urgencia": "baixo",\n'
           '  "achados": "Tosse seca de 5 dias, febre baixa (37,8°C), sem dispneia, saturação 97%.",\n'
           '  "sinais": [\n'
           '    {"sinal": "febre baixa (37,8°C)", "sistema": "outro"},\n'
           '    {"sinal": "tosse seca sem dispneia", "sistema": "respiratorio"}\n'
           "  ]\n"
           "}")


def k3_validate(p):
    if not isinstance(p, dict):
        return False, f"raiz={type(p).__name__}"
    if p.get("nivel_urgencia") not in K3_URGENCIA:
        return False, f"nivel_urgencia={p.get('nivel_urgencia')!r}"
    if not _str_ok(p.get("achados")):
        return False, "achados!=str"
    sinais = p.get("sinais")
    if not isinstance(sinais, list) or len(sinais) == 0:
        return False, "sinais vazio/ausente"
    for i, item in enumerate(sinais):
        if not isinstance(item, dict):
            return False, f"sinais[{i}]!=obj"
        if not _str_ok(item.get("sinal")):
            return False, f"sinais[{i}].sinal!=str"
        if item.get("sistema") not in K3_SISTEMAS:
            return False, f"sinais[{i}].sistema={item.get('sistema')!r}"
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
