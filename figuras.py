#!/usr/bin/env python3
"""Generate the paper figures from the real results (conformity + judge).

Writes figures/*.pdf (vector, for LaTeX) and *.png (quick preview), built entirely
from resultados/tabelas_*.json and resultados/juizo_*.json.
"""
import json
import collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- academic style, consistent across figures ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})
# Okabe-Ito (colorblind-safe). Semantics: native=fragile, fewshot=imitative, grammar=guarantee.
COL = {"native": "#D55E00", "fewshot": "#0072B2", "grammar": "#009E73"}
LAB = {"native": "native", "fewshot": "few-shot", "grammar": "grammar"}
STRATS = ["native", "fewshot", "grammar"]
CORE = ["gemma2:2b", "llama3.2:3b", "phi3:mini", "qwen2.5:3b-instruct", "qwen3:4b-instruct"]
SHORT = {"gemma2:2b": "gemma2\n2b", "llama3.2:3b": "llama3.2\n3b", "phi3:mini": "phi3\nmini",
         "qwen2.5:3b-instruct": "qwen2.5\n3b", "qwen3:4b-instruct": "qwen3\n4b"}
DOMS = [("educacao", "Education"), ("medico", "Clinical triage")]


def load(dom):
    return (json.loads((RES / f"tabelas_{dom}.json").read_text()),
            json.loads((RES / f"juizo_{dom}.json").read_text()))


def rq1_rate(tab, modelo, contrato, cond):
    for r in tab["rq1"]:
        if r["modelo"] == modelo and r["contrato"] == contrato and r["condicao"] == cond:
            return r["taxa_pct"]
    return None


def rq3_lat(tab, modelo, contrato, cond):
    for r in tab["rq3"]:
        if r["modelo"] == modelo and r["contrato"] == contrato and r["condicao"] == cond:
            return r["latencia_media_ms"]
    return None


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ok: {name}.pdf / .png")


# ============ FIG 1 -- RQ1: conformity at K3 (the headline) ============
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (dom, titulo) in zip(axes, DOMS):
        tab, _ = load(dom)
        x = np.arange(len(CORE))
        w = 0.26
        for i, s in enumerate(STRATS):
            vals = [rq1_rate(tab, m, "K3", s) or 0 for m in CORE]
            bars = ax.bar(x + (i - 1) * w, vals, w, color=COL[s], label=LAB[s], edgecolor="white", linewidth=0.4)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(100, color="#009E73", lw=0.8, ls=(0, (4, 3)), alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in CORE], fontsize=8)
        ax.set_title(titulo)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("K3 conformity (%)")
    handles = [Patch(facecolor=COL[s], label=LAB[s]) for s in STRATS]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Structural conformity on the hardest contract (K3: enum + typed list)", y=1.02, fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig1_conformity_k3")


# ============ FIG 2 -- RQ1: full surface (heatmap) ============
def fig2():
    contratos = ["K1", "K2", "K3"]
    cols = [(k, s) for k in contratos for s in STRATS]  # 9 colunas
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cmap = plt.get_cmap("RdYlGn")
    for ax, (dom, titulo) in zip(axes, DOMS):
        tab, _ = load(dom)
        M = np.full((len(CORE), len(cols)), np.nan)
        for i, m in enumerate(CORE):
            for j, (k, s) in enumerate(cols):
                v = rq1_rate(tab, m, k, s)
                if v is not None:
                    M[i, j] = v
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([LAB[s] for _, s in cols], rotation=90, fontsize=7)
        ax.set_yticks(range(len(CORE)))
        ax.set_yticklabels([m.replace(":", "\n") for m in CORE], fontsize=7)
        # contract separators
        for xline in (2.5, 5.5):
            ax.axvline(xline, color="white", lw=2)
        for j, k in enumerate(contratos):
            ax.text(1 + j * 3, -0.62, k, ha="center", va="bottom", fontsize=9, fontweight="bold")
        for i in range(len(CORE)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="black" if 25 < M[i, j] < 92 else "white")
        ax.set_title(titulo, pad=26)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("conformity (%)", fontsize=8)
    fig.suptitle("Conformity across model x contract x strategy (columns grouped by contract)", y=1.04, fontsize=11)
    savefig(fig, "fig2_conformity_heatmap")


# ============ FIG 3 -- RQ2: quality cost (blind judge) ============
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (dom, titulo) in zip(axes, DOMS):
        _, juizo = load(dom)
        data = {s: [x["nota"] for x in juizo if x["condicao"] == s and x["modelo"] in CORE] for s in STRATS}
        positions = range(1, 4)
        bp = ax.boxplot([data[s] for s in STRATS], positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1))
        for patch, s in zip(bp["boxes"], STRATS):
            patch.set_facecolor(COL[s]); patch.set_alpha(0.55); patch.set_edgecolor(COL[s])
        # mean (marker) + point jitter
        rng = np.random.default_rng(42)
        for p, s in zip(positions, STRATS):
            v = np.array(data[s])
            jit = p + (rng.random(len(v)) - 0.5) * 0.35
            ax.scatter(jit, v + (rng.random(len(v)) - 0.5) * 0.18, s=3, color=COL[s], alpha=0.25, zorder=1)
            ax.scatter([p], [v.mean()], marker="D", s=42, color="black", zorder=5)
            ax.text(p, 5.35, f"mean\n{v.mean():.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([LAB[s] for s in STRATS])
        ax.set_ylim(0.5, 5.8)
        ax.set_yticks(range(1, 6))
        ax.set_title(titulo)
    axes[0].set_ylabel("blind judge score (1-5)")
    fig.suptitle("Content quality among conformant responses (G-Eval, judge = claude-opus-4-8)", y=1.02, fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig3_quality_rq2")


# ============ FIG 4 -- RQ3: compute cost ============
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # (a) latency by strategy at K3, education (highlights the phi3 native spike)
    ax = axes[0]
    tab, _ = load("educacao")
    x = np.arange(len(CORE)); w = 0.26
    for i, s in enumerate(STRATS):
        vals = [(rq3_lat(tab, m, "K3", s) or 0) / 1000 for m in CORE]
        ax.bar(x + (i - 1) * w, vals, w, color=COL[s], label=LAB[s], edgecolor="white", linewidth=0.4)
    ax.axhline(10, color="gray", lw=0.8, ls=":")
    ax.text(len(CORE) - 0.5, 10.5, "10 s target", fontsize=7, color="gray", ha="right")
    ax.annotate("29.3 s\n0% conformant", xy=(2 - w, 29.3), xytext=(1.1, 24),
                fontsize=7, color=COL["native"], ha="center",
                arrowprops=dict(arrowstyle="->", color=COL["native"], lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in CORE], fontsize=7.5)
    ax.set_ylabel("mean latency (s)")
    ax.set_title("(a) Latency at K3 (education)")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")

    # (b) reasoning-tax: qwen3 thinking vs instruct
    ax = axes[1]
    thinking = np.mean([r["latencia_media_ms"] for r in tab["rq3"] if r["modelo"] == "qwen3:4b"]) / 1000
    instruct = np.mean([r["latencia_media_ms"] for r in tab["rq3"] if r["modelo"] == "qwen3:4b-instruct"]) / 1000
    bars = ax.bar(["qwen3:4b\n(thinking)", "qwen3:4b\n-instruct"], [thinking, instruct],
                  color=["#7B3294", "#009E73"], width=0.55, edgecolor="white")
    for b, v in zip(bars, [thinking, instruct]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f} s", ha="center", fontsize=8)
    ax.axhline(10, color="gray", lw=0.8, ls=":")
    ax.text(1.4, 11, "10 s target", fontsize=7, color="gray", ha="right")
    ax.annotate(f"~{thinking/instruct:.0f}x slower", xy=(0, thinking), xytext=(0.35, thinking * 0.7),
                fontsize=8.5, fontweight="bold")
    ax.set_ylabel("mean latency (s)")
    ax.set_title("(b) Reasoning-tax probe (education)")
    fig.suptitle("Computational cost: guaranteeing structure is not the expensive path", y=1.03, fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig4_latency_rq3")


if __name__ == "__main__":
    print("Generating figures in", OUT)
    fig1(); fig2(); fig3(); fig4()
    print("Done.")
