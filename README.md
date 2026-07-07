# SGC-SLM — The Structural-Guarantee Cost of Small Language Models

Reproducible artifact for the paper *"The Structural-Guarantee Cost of Small
Language Models: A Reproducible Protocol for Verifiable Structured Output in
Sovereign, High-Risk Domains."*

**SGC (Structural-Guarantee Cost)** is a reusable protocol that measures what it
costs to *guarantee* structured, verifiable output from a locally-run Small
Language Model (SLM, ≤4B). It compares three strategies — **native** (JSON-mode),
**few-shot** (one typed exemplar), and **grammar-constrained decoding** (typed
JSON Schema) — across a three-level contract-complexity gradient (K1 flat → K2
nested → K3 enums + typed lists), five open-weight SLMs, and two sovereign
high-risk domains (basic-education writing feedback and synthetic clinical
triage), along three axes: **conformity**, **content quality**, and **compute**.

Core finding: on the hardest contract (K3), native conformity collapses
(12–71%), few-shot fails to rescue it, and grammar-constrained decoding
guarantees it (100% at K3, ≥92% everywhere) at latency parity with — or below —
native, and with no detectable content-quality cost.

## What runs where

The conformity and compute results are **100% local** (Ollama, CPU), reproducible
by seed, with **no paid API**. The RQ2 quality scores are produced by a hosted
LLM-as-judge (`claude-opus-4-8`) used as an offline measuring instrument; the raw
judge outputs are archived under `resultados/` so RQ2 is verifiable **without
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
| Collect the matrix (per domain) | `python3 harness.py --dominio educacao` / `--dominio medico` | `resultados/conformidade[_medico].json` |
| RQ1 conformity + RQ3 compute (Fisher/Wilson) | `python3 analise.py --dominio educacao` / `medico` | `resultados/tabelas_*.json` |
| RQ2 quality (blind LLM-as-judge) | `python3 juiz.py --dominio educacao` / `medico` | `resultados/juizo_*.json`, `tabelas_juiz_*.json` |
| RQ2 equivalence (paired gap, bootstrap CI, TOST) | `python3 equivalencia_rq2.py` | printed to stdout |
| Figures 1–4 | `python3 figuras.py` | `figures/*.pdf` |

`harness.py` is incremental and **resumable** (Ctrl-C and re-run to continue).
`juiz.py` is resumable and blind to the strategy. All raw and intermediate
results are already included under `resultados/`, so the analyses above can be
re-run without re-collecting.

## Files

| File | Role |
|---|---|
| `contratos.py` | EDUCATION domain — K1/K2/K3 system prompts, JSON Schemas, few-shot exemplars, deterministic validators |
| `contratos_medico.py` | CLINICAL domain — same K1/K2/K3 for synthetic triage support |
| `harness.py` | multi-domain collector against Ollama (`--dominio`); incremental, resumable |
| `analise.py` | RQ1/RQ3 — Fisher's exact + Wilson intervals + quality proxy (`--dominio`) |
| `juiz.py` | RQ2 — blind LLM-as-judge (G-Eval pointwise 1–5), strategy-blind |
| `equivalencia_rq2.py` | RQ2 robustness — scenario-matched paired gap, bootstrap CI, TOST equivalence |
| `figuras.py` | generates the four vector figures |
| `cenarios.jsonl` | 8 student-writing scenarios (education) |
| `cenarios_medico.jsonl` | 8 synthetic clinical notes (fictitious; no real personal data) |
| `resultados/` | raw outputs (`conformidade*`, `juizo_*`) and aggregate tables (`tabelas_*`) |

## Reproducibility notes

- `temperature = 0.2`, `seed ∈ {42, 43, 44}` fixed in `harness.py`.
- Contracts, schemas, and validators are versioned in `contratos*.py`.
- Timeouts/errors count as non-conformant in the denominator (conservative).
- All scenario data is synthetic; the clinical instance is support-only (flags
  findings and asks the professional; it does not diagnose).

## License

- Code: **MIT** (see `LICENSE`).
- Data (`cenarios*.jsonl`, `resultados/`): **CC-BY 4.0**.

## Citation

If you use this artifact, please cite the paper (see `CITATION.cff`).
