# SGC-SLM: The Structural-Guarantee Cost of Small Language Models

Reproducible artifact for the paper *"The Structural-Guarantee Cost of Small
Language Models: A Reproducible Protocol for Verifiable Structured Output in
Sovereign, High-Risk Domains."*

**SGC (Structural-Guarantee Cost)** is a reusable protocol that measures what it
costs to *guarantee* structured, verifiable output from a locally-run Small
Language Model (SLM, ≤4B). It compares three strategies, **native** (JSON-mode),
**few-shot** (one typed exemplar), and **grammar-constrained decoding** (typed
JSON Schema), across a three-level contract-complexity gradient (K1 flat → K2
nested → K3 enums + typed lists), five open-weight SLMs, and two sovereign
high-risk domains (basic-education writing feedback and synthetic clinical
triage), along three axes: **conformity**, **content quality**, and **compute**.

Core finding: on the hardest contract (K3), native conformity collapses
(12–71%), few-shot fails to rescue it, and grammar-constrained decoding
guarantees it (100% at K3, ≥92% everywhere) at latency parity with (or below)
native, and with no detectable content-quality cost.

## What runs where

The conformity and compute results are **100% local** (Ollama, CPU), reproducible
by seed, with **no paid API**. The RQ2 quality scores are produced by a hosted
LLM-as-judge (`claude-opus-4-8`) used as an offline measuring instrument; the raw
judge outputs are archived under `results/` so RQ2 is verifiable **without
re-invoking the API**.

## Requirements

```bash
# 1. Ollama running locally (https://ollama.com)
ollama serve

# 2. the five core models (instruct/direct variants)
ollama pull llama3.2:3b
ollama pull qwen2.5:3b-instruct
ollama pull gemma2:2b
ollama pull phi3:mini
ollama pull qwen3:4b-instruct
# reasoning-tax probe (education only): ollama pull qwen3:4b

# 3. Python deps
pip3 install -r requirements.txt
# for the blind judge (RQ2) only:  pip3 install anthropic  &&  export ANTHROPIC_API_KEY=...
```

## Reproducing the paper

| Paper item | Command | Output |
|---|---|---|
| Collect the matrix (per domain) | `python3 harness.py --domain education` / `--domain clinical` | `results/conformity_*.json` |
| RQ1 conformity + RQ3 compute (Fisher/Wilson) | `python3 analysis.py --domain education` / `clinical` | `results/tables_*.json` |
| RQ2 quality (blind LLM-as-judge) | `python3 judge.py --domain education` / `clinical` | `results/judgments_*.json`, `tables_judge_*.json` |
| RQ2 equivalence (paired gap, bootstrap CI, TOST) | `python3 equivalence_rq2.py` | printed to stdout |
| Figures 1–4 | `python3 make_figures.py` | `figures/*.pdf` |

`harness.py` is incremental and **resumable** (Ctrl-C and re-run to continue).
`judge.py` is resumable and blind to the strategy. All raw and intermediate
results are already included under `results/`, so the analyses above can be
re-run without re-collecting.

## Files

| File | Role |
|---|---|
| `contracts_education.py` | EDUCATION domain: K1/K2/K3 system prompts, JSON Schemas, few-shot exemplars, deterministic validators |
| `contracts_clinical.py` | CLINICAL domain: same K1/K2/K3 for synthetic triage support |
| `harness.py` | multi-domain collector against Ollama (`--domain`); incremental, resumable |
| `analysis.py` | RQ1/RQ3: Fisher's exact + Wilson intervals + quality proxy (`--domain`) |
| `judge.py` | RQ2: blind LLM-as-judge (G-Eval pointwise 1–5), strategy-blind |
| `equivalence_rq2.py` | RQ2 robustness: scenario-matched paired gap, bootstrap CI, TOST equivalence |
| `make_figures.py` | generates the four vector figures |
| `scenarios_education.jsonl` | 8 student-writing scenarios (education) |
| `scenarios_clinical.jsonl` | 8 synthetic clinical notes (fictitious; no real personal data) |
| `results/` | raw outputs (`conformity_*`, `judgments_*`) and aggregate tables (`tables_*`, `tables_judge_*`) |

## Schema identifiers: Portuguese in the artifact, English in the paper

The paper names contract fields in English for readability. This artifact keeps the
original Portuguese identifiers, and the two are mapped below.

The identifiers were deliberately **not** translated in the code. The task prompts are
in Brazilian Portuguese, and the wording of a schema key is part of the stimulus the
model receives rather than an incidental label, so renaming the keys would change the
experimental condition instead of relabeling it. It would also detach the code from the
archived raw responses under `results/`, on which every number reported in the paper is
computed. Only file names, directory names, and the command-line surface were
translated; none of those appear inside the data.

**Education** (`contracts_education.py`)

| Contract | Artifact (Portuguese) | Paper (English) |
|---|---|---|
| K1 | `pontos_fortes`, `perguntas_reflexivas` | `strengths`, `reflective_questions` |
| K2 | `feedback` → `{ponto_forte, foco_koch}`, `perguntas_reflexivas` | `feedback` → `{strength, koch_focus}`, `reflective_questions` |
| K3 | `nivel_texto` *(enum)*, `pontos_fortes`, `perguntas[]` → `{pergunta, tipo (enum)}` | `text_level` *(enum)*, `strengths`, `questions[]` → `{question, type (enum)}` |

**Clinical** (`contracts_clinical.py`)

| Contract | Artifact (Portuguese) | Paper (English) |
|---|---|---|
| K1 | `achados`, `perguntas_ao_profissional` | `findings`, `questions_for_the_professional` |
| K2 | `avaliacao` → `{achado_principal, categoria}`, `perguntas_ao_profissional` | `assessment` → `{main_finding, category}`, `questions_for_the_professional` |
| K3 | `nivel_urgencia` *(enum)*, `achados`, `sinais[]` → `{sinal, sistema (enum)}` | `urgency_level` *(enum)*, `findings`, `signs[]` → `{sign, system (enum)}` |

The same applies to the judge rubrics in `judge.py` and to the quality-proxy lexicons in
`analysis.py`: both are Portuguese because they operate on Portuguese text, and both are
measurement instruments whose wording is fixed by the experiment.

Domain keys are likewise stored internally as `educacao` and `medico`, because they are
recorded inside the archived result files. Every script accepts the English names on the
command line (`--domain education`, `--domain clinical`) and the original Portuguese ones
(`--dominio educacao`, `--dominio medico`) interchangeably.

## Reproducibility notes

- `temperature = 0.2`, `seed ∈ {42, 43, 44}` fixed in `harness.py`.
- Contracts, schemas, and validators are versioned in `contracts_*.py`.
- Timeouts/errors count as non-conformant in the denominator (conservative).
- All scenario data is synthetic; the clinical instance is support-only (flags
  findings and asks the professional; it does not diagnose).

## License

- Code: **MIT** (see `LICENSE`).
- Data (`scenarios_*.jsonl`, `results/`): **CC-BY 4.0**.

## Citation

If you use this artifact, please cite the paper (see `CITATION.cff`).
