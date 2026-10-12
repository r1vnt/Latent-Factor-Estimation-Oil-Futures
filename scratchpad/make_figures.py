"""Generate the four figures article5.tex references, sized and laid out for
a SINGLE COLUMN of IEEEtran's two-column journal layout (\\columnwidth ~
3.45in), not a full page. Figures 1-2 are data-driven from real project
data; Figures 3-4 are schematics redrawn as vertical flows, since a wide
horizontal box-and-arrow layout does not survive being shrunk into a ~3.45in
column -- the earlier draft of this script got that wrong.

Run with the project's venv Python (not bare `python`).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(r"C:\Users\bradl\Documents\GitHub\Latent-Factor-Estimation-Oil-Futures")
sys.path.insert(0, str(ROOT / "src"))
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

from csr.curve.build import Panel  # noqa: E402

# IEEEtran journal class, two-column, letter paper: \columnwidth ~= 3.5in.
# Designing AT this physical width (not some larger size meant to be shrunk)
# means matplotlib fontsize values map ~1:1 onto final printed point sizes,
# so legibility can be judged directly from the rendered PNG.
COLWIDTH = 3.45

plt.rcParams.update({
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# ---------------------------------------------------------------------
# Load the real panel (same construction as every explore/*.py script)
# ---------------------------------------------------------------------
raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau, dates = panel.log_f, panel.tau, panel.dates
print(f"panel: {panel.n} dates x 18 contracts, {dates[0].date()} to {dates[-1].date()}")

# =======================================================================
# Figure 1: real backwardation and contango term structures.
# Stacked VERTICALLY (not side-by-side) so each panel keeps the full
# column width for its axis -- two narrow side-by-side panels left too
# little room for tick labels at this size.
# =======================================================================
date_back = pd.Timestamp("2013-09-06")
date_cont = pd.Timestamp("2016-02-11")
i_back = int(np.where(dates == date_back)[0][0])
i_cont = int(np.where(dates == date_cont)[0][0])

fig, axes = plt.subplots(2, 1, figsize=(COLWIDTH, 3.3), sharex=True)
for ax, i, label, color in [
    (axes[0], i_back, f"Backwardation \u2014 {date_back.date()}", "#b23b3b"),
    (axes[1], i_cont, f"Contango \u2014 {date_cont.date()}", "#2c6fa8"),
]:
    tau_i = tau[i, :] * 12.0  # years -> months
    f_i = np.exp(log_f[i, :])
    ax.plot(tau_i, f_i, "o-", color=color, markersize=2.5, linewidth=1.0)
    ax.set_title(label, fontsize=7.5, loc="left")
    ax.set_ylabel("$/bbl", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, linewidth=0.5)
axes[1].set_xlabel("Months to maturity", fontsize=7)
fig.suptitle("WTI futures term structure: two historical regimes", fontsize=8.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGS / "Figure_1.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved Figure_1.png "
     f"(C1={np.exp(log_f[i_back,0]):.2f} C18={np.exp(log_f[i_back,17]):.2f} backwardation; "
     f"C1={np.exp(log_f[i_cont,0]):.2f} C18={np.exp(log_f[i_cont,17]):.2f} contango)")

# =======================================================================
# Figure 2: GARCH(1,1) conditional volatility on front-month returns
# =======================================================================
from arch import arch_model  # noqa: E402

front = log_f[:, 0]
ret = np.diff(front)
ret = ret[np.isfinite(ret)] * 100.0  # percent log returns, arch's own convention
ret_dates = dates[1:][np.isfinite(np.diff(front))]

am = arch_model(ret, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
res = am.fit(disp="off")
print(res.summary().tables[1])
cond_vol_annual = res.conditional_volatility * np.sqrt(252) / 100.0  # back to annualised decimal vol

episodes = [
    ("2014-06-01", "2016-02-29", "2014\u201316 collapse"),
    ("2020-02-15", "2020-05-31", "COVID / negative WTI"),
    ("2022-02-01", "2022-07-31", "Ukraine invasion"),
]

fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
ax.plot(ret_dates, cond_vol_annual, color="#1a1a1a", linewidth=0.6)
for start, end, label in episodes:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#d9534f", alpha=0.18)
ax.set_ylabel("Ann. cond. vol.", fontsize=7)
ax.tick_params(labelsize=6.5)
ax.set_title("GARCH(1,1) conditional volatility, WTI front-month returns", fontsize=7.5, loc="left")
ax.grid(alpha=0.25, linewidth=0.5)
handles = [mpatches.Patch(color="#d9534f", alpha=0.18, label=lbl) for _, _, lbl in episodes]
ax.legend(handles=handles, fontsize=5.8, loc="upper left", frameon=False, handlelength=1.2,
         labelspacing=0.3, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(FIGS / "figure_2.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"saved figure_2.png (GARCH params: {dict(res.params.round(4))})")


# =======================================================================
# Shared helper for the two schematic figures
# =======================================================================
def box(ax, x, y, w, h, text, fc="#eef2f7", ec="#333333", fontsize=6.3, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.05",
                       linewidth=0.9, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
           fontweight="bold" if bold else "normal", linespacing=1.4)
    return b


def arrow(ax, p0, p1, **kw):
    kw.setdefault("arrowstyle", "-|>")
    kw.setdefault("linewidth", 1.1)
    kw.setdefault("color", "#333333")
    kw.setdefault("mutation_scale", 9)
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


# =======================================================================
# Figure 3: news-augmented framework -- CONCEPTUAL, Part-I design diagram,
# redrawn as a VERTICAL flow to fit a single column. Neutral/symmetric per
# the existing caption (the driver/measurement asymmetry is a Part-II
# result reported in prose in sec:newsresult, not drawn here).
# =======================================================================
fig, ax = plt.subplots(figsize=(COLWIDTH, 3.75))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13.0)
ax.axis("off")

# row 1: two input sources
box(ax, 0.3, 10.8, 4.3, 1.7, "Textual news\n(headlines, articles)", fc="#fdf3e7")
box(ax, 5.4, 10.8, 4.3, 1.7, "Observed futures prices\n(across maturities)", fc="#eef2f7")

# row 2: processing
box(ax, 0.3, 8.1, 4.3, 1.9, "NLP models\nembeddings, sentiment,\ntopic indicators", fc="#fdf3e7")
box(ax, 5.4, 8.1, 4.3, 1.9, "Price-based\ninformation", fc="#eef2f7")

arrow(ax, (2.45, 10.8), (2.45, 10.0))
arrow(ax, (7.55, 10.8), (7.55, 10.0))

# row 3: the two latent factors (single combined box, column too narrow for two)
box(ax, 0.3, 4.1, 9.4, 2.9,
   "Latent factors of the two-factor model\n"
   "$\\chi_t$: temporary deviations (inventory effects, shocks)\n"
   "$\\xi_t$: equilibrium expectations, structural outlook",
   fc="#e8f0e4", fontsize=6.3)

arrow(ax, (2.45, 8.1), (3.8, 7.0), connectionstyle="arc3,rad=-0.15")
arrow(ax, (7.55, 8.1), (6.2, 7.0), connectionstyle="arc3,rad=0.15")

# row 4: inference
box(ax, 1.2, 1.1, 7.6, 1.8, "Inference on the term structure\nwithin the two-factor framework", fc="#f6f6f6", ec="#555555")
arrow(ax, (5.0, 4.1), (5.0, 2.9))

ax.text(5.0, 0.55, "Both channels are hypothesised to inform both factors;\n"
                   "a later section asks whether each drives the factors or measures them.",
       fontsize=5.7, color="#444444", ha="center", va="center", style="italic")

ax.set_title("Conceptual framework: news-augmented latent factor modelling",
            fontsize=8, pad=8, wrap=True)
fig.tight_layout()
fig.savefig(FIGS / "Figure_3.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved Figure_3.png")

# =======================================================================
# Figure 4: state-space representation, redrawn as a VERTICAL chain
# (state at t-1 -> state at t -> observed prices), correctly labelled
# chi_t/xi_t throughout, with the transition and measurement equations
# as arrow labels rather than separate boxes -- compact enough for a
# single column without crowding either equation.
# =======================================================================
fig, ax = plt.subplots(figsize=(COLWIDTH, 3.05))
ax.set_xlim(0, 10)
ax.set_ylim(0, 9.5)
ax.axis("off")

box(ax, 0.6, 7.5, 8.8, 1.7, "State at $t-1$:  $\\chi_{t-1},\\ \\xi_{t-1}$", fc="#e8f0e4", fontsize=7)
box(ax, 0.6, 3.85, 8.8, 1.7, "State at $t$:  $\\chi_t,\\ \\xi_t$", fc="#e8f0e4", fontsize=7)
box(ax, 0.6, 0.2, 8.8, 1.7, "Observed log futures $\\mathbf{y}_t$\n(18 contracts)", fc="#eef2f7", fontsize=7)

arrow(ax, (5.0, 7.5), (5.0, 5.45), linewidth=1.4, mutation_scale=11)
ax.text(5.0, 6.65, "Transition", fontsize=6.6, ha="center", va="center", fontweight="bold",
       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
ax.text(5.0, 6.1, "$\\mathbf{T}\\mathbf{x}_{t-1}+\\mathbf{c}+\\mathbf{w}_t$", fontsize=6.3,
       ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

arrow(ax, (5.0, 3.85), (5.0, 1.8), linewidth=1.4, mutation_scale=11)
ax.text(5.0, 3.0, "Measurement", fontsize=6.6, ha="center", va="center", fontweight="bold",
       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
ax.text(5.0, 2.45, "$\\mathbf{y}_t=\\mathbf{d}_t+\\mathbf{Z}_t\\mathbf{x}_t+\\mathbf{v}_t$", fontsize=6.3,
       ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

ax.set_title("State-space representation of the two-factor commodity pricing model",
            fontsize=8, pad=8, wrap=True)
fig.tight_layout()
fig.savefig(FIGS / "Figure_4.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved Figure_4.png")

print("\nAll four figures written to", FIGS, f"(designed at column width {COLWIDTH}in)")
