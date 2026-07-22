#!/usr/bin/env python3
"""Paper-grade figures for the observed→latent de-biasing (bead slr-5ol).

Renders the numeric findings (slr-0rn/bm3/60q) into the paper's visuals as vector
PDFs → `doc/assets/`. Tells the honest story: de-biasing corrects magnitude and
composition, it does NOT reorder the map. Everything is pooled over the declared
panel **2019–2024** and labelled in **English**.

  Fig 1  fig-desesgo-cifranegra: two panels — (a) direct pooled reporting rate r̂
         per category with 80% Jeffreys CI; (b) pooled dark-figure multiplier
         Σλ*/Σy per category on a log axis with an uncertainty band.
  Fig 2  fig-desesgo-mapas: three Lima maps — observed | latent | under-
         representation (latent − observed), robbery only, annual-mean rate/100k.
  Fig 3  fig-desesgo-ranking: observed vs latent district rank, robbery only.
  Fig 4  fig-desesgo-trust: district PNP-trust vs robbery multiplier.
  (fig-desesgo-composicion is generated too but is not used in the paper.)

Stack exception (scoped to figures): matplotlib + geopandas + scipy (not just
numpy/scipy), like the viz. No S2, no network.

Usage:  python3 scripts/figures_desesgo.py
Output: doc/assets/fig-desesgo-{composicion,cifranegra,mapas,ranking,trust}.pdf
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, spearmanr

mpl.use("Agg")
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

ROOT = Path(__file__).resolve().parent.parent
LS = ROOT / "data" / "datasets" / "silver" / "latent_surface"
LAT = LS / "latente_distrito_categoria_anio.csv"
VAL = LS / "validacion.json"
RDIR = ROOT / "data" / "datasets" / "silver" / "reporting_rate" / "r_directo_categoria_anio.csv"
POB = ROOT / "data" / "datasets" / "silver" / "inei_poblacion" / "poblacion_distrital_2018_2026.csv"
SHP = ROOT / "data" / "datasets" / "DISTRITOS_LIMITES.zip"
TRUST_JSON = ROOT / "data" / "silver" / "analysis" / "district_trust_pnp.json"
# Composición canónica (4 cats, excl. violencia familiar) — la emite build_composition_canon.py.
COMP = ROOT / "data" / "silver" / "analysis" / "composition_excl_dv.json"
ASSETS = ROOT / "doc" / "assets"

# Declared panel: MININTER/ENAPRES 2019–2024 (2018 excluded, corrupt UBIGEO).
Y0, Y1 = 2019, 2024
YEARS = range(Y0, Y1 + 1)

# English labels, keyed by the canonical category code used across artifacts.
LABELS = {"robo_hurto_callejero": "Robbery & mugging", "secuestro": "Kidnapping",
          "extorsion": "Extortion", "violencia_familiar_sexual": "Domestic violence",
          "estafa": "Fraud & scams"}
# r_directo.csv uses the same category codes for these five; robbery = only 2024 wave.
RDIR_CATS = ["robo_hurto_callejero", "secuestro", "extorsion",
             "violencia_familiar_sexual", "estafa"]
UNSTABLE = {"estafa"}          # fraud: r̂ ≈ 0.017, multiplier flagged unstable
ROBBERY = "robo_hurto_callejero"
SHORT_NAME = {"150143": "V.M. Triunfo", "150140": "Surco", "150112": "Independencia"}

ORANGE, GREY, TEAL, BLUE = "#d95f0e", "#bdbdbd", "#7fcdbb", "#2c7fb8"


# ── loaders ─────────────────────────────────────────────────────────────────────
def load_latent():
    rows = list(csv.DictReader(open(LAT, encoding="utf-8")))
    out = []
    for r in rows:
        r["anio"] = int(r["anio"])
        if not (Y0 <= r["anio"] <= Y1):        # HARD panel filter — drops 2018
            continue
        for k in ("observado", "latente", "latente_ic_low", "latente_ic_high",
                  "factor_subrep", "r_hat"):
            r[k] = float(r[k]) if r[k] not in ("", "nan", "inf") else float("nan")
        r["inestable"] = int(r["inestable"])
        out.append(r)
    return out


def load_pop():
    p = {}
    for r in csv.DictReader(open(POB, encoding="utf-8")):
        p[(r["ubigeo"], int(r["anio"]))] = int(r["poblacion"])
    return p


def pooled_reporting_rate():
    """Direct victim-level r̂ pooled over LIMA, n_eff-weighted, with 80% Jeffreys CI
    (Beta(k+0.5, n−k+0.5), quantiles 0.10/0.90, k=Σr·n_eff, n=Σn_eff)."""
    agg = defaultdict(lambda: [0.0, 0.0])      # cat -> [k=Σr·n, n=Σn]
    for r in csv.DictReader(open(RDIR, encoding="utf-8")):
        if r["ambito"] != "LIMA" or r["categoria"] not in RDIR_CATS:
            continue
        yr = int(r["anio"])
        if not (Y0 <= yr <= Y1):
            continue
        rv, n = float(r["r_pnp_victima"]), float(r["n_eff"])
        agg[r["categoria"]][0] += rv * n
        agg[r["categoria"]][1] += n
    out = {}
    for c, (k, n) in agg.items():
        rhat = k / n
        lo = beta.ppf(0.10, k + 0.5, n - k + 0.5)
        hi = beta.ppf(0.90, k + 0.5, n - k + 0.5)
        out[c] = dict(rhat=rhat, lo=lo, hi=hi)
    return out


def pooled_multiplier(rows):
    """Pooled dark-figure multiplier Σλ*/Σy per category, with band
    [Σλ*_low/Σy, Σλ*_high/Σy]. If any row has an infinite/NaN upper λ* (r_ic_low=0),
    the upper band is unbounded → flagged truncated=True."""
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, False, 0])
    # cat -> [Σobs, Σlat, Σlat_low, Σlat_high, upper_truncated, inestable]
    for r in rows:
        c = r["categoria"]
        g = agg[c]
        g[0] += r["observado"]
        g[1] += r["latente"]
        lo = r["latente_ic_low"]
        hi = r["latente_ic_high"]
        g[2] += lo if lo == lo else 0.0
        if (hi != hi) or math.isinf(hi):       # NaN or inf upper bound
            g[4] = True
        else:
            g[3] += hi
        g[5] |= r["inestable"]
    out = {}
    for c, (obs, lat, lo, hi, trunc, ie) in agg.items():
        if obs <= 0:
            continue
        out[c] = dict(mult=lat / obs, band_lo=lo / obs,
                      band_hi=(hi / obs), truncated=trunc, inestable=bool(ie))
    return out


# ── Fig 1 — reporting rates + dark-figure multipliers ───────────────────────────
def fig_cifranegra(rows):
    rr = pooled_reporting_rate()
    mm = pooled_multiplier(rows)
    # Vertical order shared by both panels: r̂ descending (tie → larger n via rhat).
    order = sorted(rr, key=lambda c: rr[c]["rhat"], reverse=True)
    y = np.arange(len(order))[::-1]            # top row = highest r̂

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # (a) direct pooled reporting rate with 80% Jeffreys CI — points, not bars
    for yi, c in zip(y, order):
        d = rr[c]
        axL.errorbar(d["rhat"], yi, xerr=[[d["rhat"] - d["lo"]], [d["hi"] - d["rhat"]]],
                     fmt="o", ms=6, color=ORANGE, ecolor="#555", elinewidth=1.1,
                     capsize=3, zorder=3)
    axL.set_yticks(y)
    axL.set_yticklabels([LABELS[c] for c in order])
    axL.set_xlim(0, 0.32)
    axL.set_xlabel("Reporting rate to PNP, r̂ (80% CI)")
    axL.set_title("(a) Direct reporting rate", fontsize=9)
    axL.grid(axis="x", color="#eee", zorder=0)

    # (b) pooled multiplier, log x, with uncertainty band; fraud grey + "unstable"
    xmax = 1600.0
    for yi, c in zip(y, order):
        d = mm[c]
        col = GREY if c in UNSTABLE else ORANGE
        # lower whisker
        axR.plot([d["band_lo"], d["mult"]], [yi, yi], color="#555", lw=1.1, zorder=2)
        if d["truncated"]:                     # unbounded upper → arrow to edge + "›"
            axR.plot([d["mult"], xmax], [yi, yi], color="#555", lw=1.1, zorder=2)
            axR.annotate("›", (xmax, yi), fontsize=13, color="#555",
                         ha="right", va="center", zorder=4)
        else:
            axR.plot([d["mult"], d["band_hi"]], [yi, yi], color="#555", lw=1.1, zorder=2)
        axR.plot(d["mult"], yi, "o", ms=6, color=col, zorder=3)
        note = f"×{d['mult']:.1f}" + ("  unstable" if c in UNSTABLE else "")
        axR.annotate(note, (d["mult"], yi + 0.16), fontsize=7.5,
                     color=("#666" if c in UNSTABLE else "#333"), ha="center", va="bottom")
    axR.set_xscale("log")
    axR.set_xlim(1.3, xmax)
    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.set_ylim(-0.6, len(order) - 0.4)
    axL.set_ylim(-0.6, len(order) - 0.4)
    axR.set_xlabel("Dark-figure multiplier  Σλ*/Σy  (log scale)")
    axR.set_title("(b) Under-reporting multiplier", fontsize=9)
    axR.grid(axis="x", which="both", color="#eee", zorder=0)

    fig.suptitle("Reporting rates and dark-figure multipliers, Lima 2019–2024",
                 fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig-desesgo-cifranegra")


# ── Fig 2 — observed / latent / under-representation maps (robbery) ──────────────
def fig_mapas(rows, pop):
    gdf = gpd.read_file(f"zip://{SHP}")
    gdf = gdf[gdf["UBIGEO"].astype(str).str.startswith("1501")].copy()
    gdf["ubigeo"] = gdf["UBIGEO"].astype(str)
    obs, lat = defaultdict(float), defaultdict(float)
    for r in rows:
        if r["categoria"] == ROBBERY:          # already 2019–2024 only
            obs[r["ubigeo"]] += r["observado"]
            lat[r["ubigeo"]] += r["latente"]
    py = defaultdict(float)                     # person-years, 2019–2024
    for u in set(obs) | set(lat) | {g for g in gdf["ubigeo"]}:
        py[u] = sum(pop.get((u, yr), 0) for yr in YEARS)

    def rate(src, u):                           # annual-mean rate per 100k
        return src.get(u, 0.0) / py[u] * 1e5 if py.get(u) else np.nan

    gdf["obs_r"] = [rate(obs, u) for u in gdf["ubigeo"]]
    gdf["lat_r"] = [rate(lat, u) for u in gdf["ubigeo"]]
    gdf["gap_r"] = gdf["lat_r"] - gdf["obs_r"]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.0))
    panels = [("obs_r", "Observed (reports)", "Oranges"),
              ("lat_r", "Latent (de-biased)", "Reds"),
              ("gap_r", "Under-representation\n(latent − observed)", "Purples")]
    for ax, (col, title, cmap) in zip(axes, panels):
        gdf.plot(column=col, cmap=cmap, linewidth=0.2, edgecolor="white", ax=ax,
                 legend=True, legend_kwds={"shrink": 0.5, "label": "per 100k / year"},
                 missing_kwds={"color": "lightgrey"})
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Observed vs. latent robbery risk, Lima 2019–2024 "
                 "(annual mean rate per 100,000)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig-desesgo-mapas")


# ── Fig 3 — ranking stability (robbery) ─────────────────────────────────────────
def fig_ranking(rows):
    obs, lat = defaultdict(float), defaultdict(float)
    for r in rows:
        if r["categoria"] == ROBBERY:          # 2019–2024, robbery only
            obs[r["ubigeo"]] += r["observado"]
            lat[r["ubigeo"]] += r["latente"]
    us = sorted(set(obs) | set(lat))
    ov = np.array([obs.get(u, 0) for u in us])
    lv = np.array([lat.get(u, 0) for u in us])
    ro = len(us) - np.argsort(np.argsort(ov))  # rank 1 = highest
    rl = len(us) - np.argsort(np.argsort(lv))
    rho = spearmanr(ov, lv).statistic
    shifts = np.abs(ro - rl)
    # ρ(obs, lat) de robo es portante (§5.4 caption + prosa): el 0.995 hand-copiado quedó
    # stale cuando la superficie se regeneró (real 0.992) — emitir lo bloquea (slr-x50g v2).
    import canon
    canon.emit(
        "rank_rho_obs_lat.robo_hurto_callejero", float(rho), variant="victim",
        unit="Spearman entre ranking distrital observado y latente de robo (43 distritos, pooled)",
        estimator="conteos pooled 2019–2024 de la superficie, rank 1 = mayor",
        inputs=[LAT], script=__file__,
    )

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([1, len(us)], [1, len(us)], "--", color="grey", lw=0.8, zorder=0)
    ax.scatter(ro, rl, s=24, color=ORANGE, alpha=0.85, edgecolor="white",
               linewidth=0.4, zorder=2)
    for u, x, yv, s in zip(us, ro, rl, shifts):
        if s >= 3:
            ax.annotate(SHORT_NAME.get(u, u[-3:]), (x, yv), fontsize=6.5,
                        xytext=(4, 4), textcoords="offset points", zorder=3)
    ax.set_xlabel("Observed rank (reports)")
    ax.set_ylabel("Latent rank (de-biased)")
    ax.set_title(f"De-biasing barely reorders the map\n"
                 f"Spearman ρ = {rho:.3f} (max shift {int(shifts.max())} "
                 f"of {len(us)})", fontsize=9.5)
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "fig-desesgo-ranking")


# ── Fig 4 — district PNP-trust vs robbery multiplier ────────────────────────────
def fig_trust():
    if not TRUST_JSON.exists():
        print(f"⚠ skip fig-desesgo-trust: missing {TRUST_JSON.relative_to(ROOT)} "
              f"(run scripts/build_district_trust.py)")
        return
    d = json.load(open(TRUST_JSON, encoding="utf-8"))
    pdist = d["per_district"]
    x = np.array([v["trust"] * 100 for v in pdist.values()])
    y = np.array([v["robbery_multiplier"] for v in pdist.values()])
    rho, p, n = d["spearman_rho"], d["spearman_p"], d["n_districts"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(x, y, s=26, color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.4)
    b, a = np.polyfit(x, y, 1)                  # linear trend (visual only)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a + b * xs, color="#555", lw=1, ls="--")
    ax.set_xlabel("District-level trust in the PNP "
                  "(share ≥ 'sufficient', ENAPRES 2022–2023)")
    ax.set_ylabel("Robbery dark-figure multiplier")
    ax.set_title(f"Less trust, larger reporting gap\n"
                 f"Spearman ρ = {rho:.2f} (p = {p:.3f}, n = {n})", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig-desesgo-trust")


# ── Fig 5 (secondary, not used in paper) — composition share ────────────────────
def fig_composicion():
    """Composición por tipo, excl. violencia familiar (universos no comparables, §3.1).

    Lee el artefacto del generador canónico, NO `validacion.json` directamente: la
    renormalización sobre las cuatro categorías comparables y la elección de base ENAPRES
    (anualizada, ponderada por olas) viven en `build_composition_canon.py`, que las emite
    al registro. Duplicar ese cálculo aquí es cómo se coló el titular falso L1 0.99→0.11.
    """
    d = json.load(open(COMP, encoding="utf-8"))
    cats, sh = d["cats"], d["shares"]
    base = d["canonical_base"]
    o, l, v = (np.array(sh[k]) for k in ("observed", "latent", f"enapres_{base}"))
    order = np.argsort(-v)
    cats = [cats[i] for i in order]
    o, l, v = o[order], l[order], v[order]
    yy = np.arange(len(cats))
    h = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(yy + h, v * 100, h, label="ENAPRES victimisation (benchmark)", color=BLUE)
    ax.barh(yy, l * 100, h, label="Latent (de-biased)", color=TEAL)
    ax.barh(yy - h, o * 100, h, label="Observed (reports)", color=ORANGE)
    ax.set_yticks(yy)
    ax.set_yticklabels([LABELS[c] for c in cats])
    ax.set_xlabel("Share of total crime (%)")
    l1 = d["l1"][base]
    ax.set_title("Crime composition: de-biasing moves AWAY from the survey mix\n"
                 f"(L1 to ENAPRES: observed {l1['observed']:.3f} → "
                 f"latent {l1['latent']:.3f})", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "fig-desesgo-composicion")


def save(fig, name):
    dest = ASSETS / f"{name}.pdf"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {dest.relative_to(ROOT)}")


def main():
    rows = load_latent()
    pop = load_pop()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_composicion()
    fig_cifranegra(rows)
    fig_mapas(rows, pop)
    fig_ranking(rows)
    fig_trust()
    print("\nFigures regenerated in doc/assets/ — English, pooled 2019–2024. "
          "Honest story: magnitude within robbery, NOT composition (the correction moves the "
          "type mix away from the survey) and NOT spatial reordering.")


if __name__ == "__main__":
    main()
