#!/usr/bin/env python3
"""Generate the SIMBig 2026 manuscript figures (slr-p1u.11).

Reproducible from versioned artifacts + silver data; no notebooks, no
external assets. Outputs vector PDF to doc/assets/ (versioned).

Figures
-------
fig-simbig-maps.pdf         Oracle vs prediction, street robbery 2023.
                            Lima row: geocoded oracle vs tabular-ceiling M4c
                            (protocol = eval_modality_ladder.py, same HGB
                            params, train 2019-2022). Arequipa row: own oracle
                            vs Lima-trained 18-feature transfer (protocol =
                            transfer_arequipa.py run_eval).
fig-simbig-degradation.pdf  Evaluability curve from
                            data/silver/predictions/selective_geocoding_degradation.csv
                            (uniform + selective loss; report:
                            analysis/selective_geocoding_degradation.md).
fig-simbig-learning.pdf     Local-vs-transfer paired deltas with 95% CIs from
                            analysis/etapa2_learning_curve.md
                            (exp_learning_curve.py, B=2000).

--lang es emits the same figures with Spanish in-figure text under a "-es"
suffix, for doc/ccis/simbig-ccis-es.tex. Default (en) output is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "doc/assets"
# One run supplies both arms (uniform = MCAR upper bound, selective = empirical
# propensity), so table and figure come from the same seeds (slr-7lsf).
DEGRADATION_CSV = ROOT / "data/silver/predictions/selective_geocoding_degradation.csv"
AQP_ORACLE = ROOT / "data/silver/transfer_arequipa/aqp_oracle.parquet"
AQP_FEATURES = ROOT / "data/silver/transfer_arequipa/aqp_features.parquet"

CAT = "robo_hurto_callejero"  # highest-volume category: most legible map
TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023

# Okabe-Ito subset; validated (dataviz six checks, light surface):
# CVD worst adjacent-pair dE 37.2, all contrast/lightness/chroma PASS.
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_GREEN = "#009E73"
C_GRID = "#d9d9d9"
C_TEXT = "#333333"

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": C_GRID,
    "axes.linewidth": 0.6,
    "grid.color": C_GRID,
    "grid.linewidth": 0.5,
}

# In-figure text, by language. Nothing here is a load-bearing number: the
# figures read their values from the artifacts, these are only labels.
LABELS = {
    "en": {
        "lima_oracle": "Lima — geocoded reference 2023",
        "lima_pred": "Lima — tabular ceiling",
        "aqp_oracle": "Arequipa — own reference 2023",
        "aqp_pred": "Arequipa — Lima-trained transfer",
        "cbar": "complaints, log(1+n)",
        "skill_full": "Full-reference skill",
        "skill_degraded": "Degraded-reference skill",
        "persistence": "Persistence baseline",
        "sel_suffix": " (selective loss)",
        "uni_suffix": " (uniform loss)",
        "trujillo": "Trujillo-like",
        "geo_rate": "Geocoding rate (%)",
        "rho_macro": "Intra-district Spearman ρ (macro)",
        "parity": "transfer parity (Lima-trained, no local data)",
        "tick_year": "{y}",
        "tick_years": "{y}",
        "history": "Local Arequipa training history (years)",
        "delta_rho": "Δ macro ρ\n(local − transfer)",
    },
    "es": {
        "lima_oracle": "Lima — referencia geocodificada 2023",
        "lima_pred": "Lima — techo tabular",
        "aqp_oracle": "Arequipa — referencia propia 2023",
        "aqp_pred": "Arequipa — transferencia desde Lima",
        "cbar": "denuncias, log(1+n)",
        "skill_full": "Referencia completa",
        "skill_degraded": "Referencia degradada",
        "persistence": "Persistencia",
        "sel_suffix": " (pérdida selectiva)",
        "uni_suffix": " (pérdida uniforme)",
        "trujillo": "tipo Trujillo",
        "geo_rate": "Tasa de geocodificación (%)",
        "rho_macro": "Spearman ρ intra-distrital (macro)",
        "parity": "paridad con la transferencia (Lima, sin datos locales)",
        "tick_year": "{y}",
        "tick_years": "{y}",
        "history": "Historia local de entrenamiento, años (Arequipa)",
        "delta_rho": "Δ macro ρ\n(local − transferencia)",
    },
}
LANG = "en"


def L(key: str) -> str:
    return LABELS[LANG][key]


def out_path(stem: str) -> Path:
    """English keeps the canonical name; other languages get a suffix."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{stem}{'' if LANG == 'en' else f'-{LANG}'}.pdf"


def h3_polygon(cell: str) -> Polygon:
    boundary = h3.cell_to_boundary(cell)
    return Polygon([(lng, lat) for lat, lng in boundary])


def fit_ladder_hgb(train: pd.DataFrame, feats: list[str]):
    """Same HGB configuration as eval_modality_ladder.py / transfer_arequipa.py."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    med = train[feats].median(numeric_only=True)
    model = HistGradientBoostingRegressor(
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        learning_rate=0.06,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(train[feats].fillna(med).to_numpy(), np.log1p(train["target"].to_numpy(float)))
    return model, med


def predict_counts(model, med, frame: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return np.expm1(np.maximum(model.predict(frame[feats].fillna(med).to_numpy()), 0.0))


def to_gdf(frame: pd.DataFrame, epsg: str) -> gpd.GeoDataFrame:
    frame = frame.copy()
    frame["geometry"] = frame["h3_index"].map(h3_polygon)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326").to_crs(epsg)


def build_lima_panels() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Lima M4c ceiling model: reuse the ladder's own panel and feature set."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_modality_ladder import FEATURE_SETS, available, load_panel

    panel, _ = load_panel()
    panel = panel[panel["crime_cat"] == CAT]
    feats = available(FEATURE_SETS["M4c_transporte"], panel)
    train = panel[panel["year"].isin(TRAIN_YEARS)]
    test = panel[panel["year"] == TEST_YEAR].copy()
    return train, test, feats


def build_aqp_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Arequipa transfer map: Lima-trained transferable block (18 features)."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_modality_ladder import load_panel
    from transfer_arequipa import TRANSFER_FEATS

    feats_aqp = pd.read_parquet(AQP_FEATURES)
    oracle = pd.read_parquet(AQP_ORACLE)
    oracle = oracle[(oracle["crime_cat"] == CAT) & (oracle["year"] == TEST_YEAR)]
    obs = oracle.groupby("h3_index", as_index=False)["obs_geo_count"].sum()

    panel_lima, _ = load_panel()
    panel_lima = panel_lima[panel_lima["crime_cat"] == CAT]
    feats = [f for f in TRANSFER_FEATS if f in panel_lima.columns and f in feats_aqp.columns]

    train = panel_lima[panel_lima["year"].isin(TRAIN_YEARS)]
    test = feats_aqp.merge(obs, on="h3_index", how="left")
    test["obs_geo_count"] = test["obs_geo_count"].fillna(0.0)
    # same cell filter as transfer_arequipa.run_eval: keep populated/known-
    # district cells, dropping the empty bbox frame outside the urban area
    known = set(pd.read_parquet(AQP_ORACLE)["h3_index"])
    test = test[test["h3_index"].isin(known) | (test["population"].fillna(0) > 0)].copy()
    return train, test, feats


def fig_maps() -> Path:
    lima_train, lima_test, lima_feats = build_lima_panels()
    m_lima, med_lima = fit_ladder_hgb(lima_train, lima_feats)
    lima_test["pred"] = predict_counts(m_lima, med_lima, lima_test, lima_feats)

    aqp_train, aqp_test, aqp_feats = build_aqp_frames()
    m_aqp, med_aqp = fit_ladder_hgb(aqp_train, aqp_feats)
    aqp_test["pred"] = predict_counts(m_aqp, med_aqp, aqp_test, aqp_feats)

    gdf_lima = to_gdf(lima_test[["h3_index", "target", "pred"]], "EPSG:32718")
    aqp_test = aqp_test.rename(columns={"obs_geo_count": "target"})
    gdf_aqp = to_gdf(aqp_test[["h3_index", "target", "pred"]], "EPSG:32719")

    mpl.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.6))
    rows = [
        (gdf_lima, L("lima_oracle"), L("lima_pred")),
        (gdf_aqp, L("aqp_oracle"), L("aqp_pred")),
    ]
    # single-hue sequential, light -> dark: zero cells stay near-white
    # (print-friendly, grayscale-legible, lightness-monotonic)
    cmap = "Blues"
    for r, (gdf, t_obs, t_pred) in enumerate(rows):
        vmax = np.log1p(pd.concat([gdf["target"], gdf["pred"]]).astype(float)).quantile(0.995)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=max(vmax, 1.0))
        for c, col, title in ((0, "target", t_obs), (1, "pred", t_pred)):
            ax = axes[r, c]
            gdf.assign(v=np.log1p(gdf[col].astype(float))).plot(
                column="v", cmap=cmap, norm=norm, linewidth=0.05,
                edgecolor="white", ax=ax,
            )
            ax.set_title(title)
            ax.set_axis_off()
            ax.set_aspect("equal")
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=list(axes[r, :]), shrink=0.72, pad=0.02)
        cbar.set_label(L("cbar"), fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    out = out_path("fig-simbig-maps")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_degradation() -> Path:
    df = pd.read_csv(DEGRADATION_CSV)

    def agg_of(mech: str) -> pd.DataFrame:
        a = df[df["mechanism"] == mech].groupby("rate").agg(
            medible=("medible", "mean"), medible_sd=("medible", "std"),
            real=("real", "mean"), real_sd=("real", "std"),
            persist=("persist", "mean"), persist_sd=("persist", "std"),
        ).reset_index().sort_values("rate")
        a["x"] = a["rate"] * 100
        return a

    uni, sel = agg_of("uniforme"), agg_of("selectivo")

    mpl.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(4.9, 3.2))
    # Uniform arm: filled markers, sd band. Selective arm: same colour, hollow
    # markers, dash-dot, no band — the eye pairs each curve with its MCAR bound.
    # Full-reference skill is drawn once (uniform): the selective arm moves it
    # by ≤0.013 and would only clutter (the text reports that delta).
    series = [
        (uni, "real", L("skill_full"), C_BLUE, "-", "o", True, 8),
        (uni, "medible", L("skill_degraded") + L("uni_suffix"), C_VERMILLION, "--", "s", True, 6),
        (sel, "medible", L("skill_degraded") + L("sel_suffix"), C_VERMILLION, "-.", "s", False, -8),
        (uni, "persist", L("persistence") + L("uni_suffix"), C_GREEN, ":", "^", True, 6),
        (sel, "persist", L("persistence") + L("sel_suffix"), C_GREEN, "-.", "^", False, -8),
    ]
    for a, col, label, color, ls, marker, filled, dy in series:
        kw = dict(markersize=4, linewidth=1.5, label=label)
        if not filled:
            kw.update(markerfacecolor="white", markeredgecolor=color)
        ax.plot(a["x"], a[col], ls, color=color, marker=marker, **kw)
        if filled:
            sd = a[f"{col}_sd"].fillna(0)
            ax.fill_between(a["x"], a[col] - sd, a[col] + sd,
                            color=color, alpha=0.15, linewidth=0)

    # city markers labelled above the axes: outside the data area, so they
    # can never collide with the legend or the series labels
    for rate, name in ((83.4, "Lima"), (24.9, L("trujillo"))):
        ax.axvline(rate, color=C_GRID, linewidth=0.8, zorder=0)
        ax.annotate(name, (rate, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=6.5, color=C_TEXT)

    ax.set_xlabel(L("geo_rate"))
    ax.set_ylabel(L("rho_macro"))
    ax.set_xlim(5, 90)
    ax.grid(axis="y", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)

    out = out_path("fig-simbig-degradation")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_learning_curve() -> Path:
    # Paired deltas local-minus-transfer, macro rho, test 2023, bootstrap over
    # districts B=2000. Source of record: analysis/etapa2_learning_curve.md
    # (exp_learning_curve.py); plotted as deltas so the figure carries the
    # inferential claim (CI vs 0), not raw levels.
    years = [1, 2, 3, 4]
    delta = [0.0146, 0.0428, 0.0491, 0.0528]
    lo = [-0.0322, 0.0028, 0.0176, 0.0164]
    hi = [0.0602, 0.0824, 0.0841, 0.0875]
    sig = [False, True, True, True]

    # Esta figura se inserta a 0.38\textwidth (~1.83 in de ancho impreso). Con el
    # lienzo de 4.4 in el factor de escala era 0.42, así que las etiquetas de 7 pt
    # se imprimían a ~2.9 pt. Se conserva el aspect ratio 4.4:3.0 y el ancho en la
    # página, pero sobre un lienzo menor: escala 1.83/2.6 = 0.70, de modo que
    # 10--11 pt aquí caen en los 7--8 pt efectivos. El alto 2.58 se eligió
    # numéricamente para que el bbox recortado conserve el aspect ratio original
    # (1.436): sin las etiquetas de valor, 1.85 da 1.434.
    mpl.rcParams.update(STYLE)
    mpl.rcParams.update({
        "axes.labelsize": 11,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(2.6, 2.28))
    ax.axhline(0.0, color=C_TEXT, linewidth=1.0)
    ax.annotate(L("parity"), (0.60, 0.0),
                xytext=(0, -8), textcoords="offset points", fontsize=9,
                color=C_TEXT, va="top")
    for xi, d, l, h, s in zip(years, delta, lo, hi, sig):
        ax.errorbar(xi, d, yerr=[[d - l], [h - d]], fmt="o", markersize=5,
                    color=C_BLUE, markerfacecolor=C_BLUE if s else "white",
                    markeredgecolor=C_BLUE, capsize=2.5, linewidth=1.2)


    ax.set_xticks(years)
    ax.set_xticklabels([
        L("tick_year" if y == 1 else "tick_years").format(y=y) for y in years
    ])
    ax.set_xlabel(L("history"))
    ax.set_ylabel(L("delta_rho"))
    ax.set_xlim(0.5, 4.5)
    ax.grid(axis="y", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    mpl.rcParams.update(STYLE)  # deshace el override local
    out = out_path("fig-simbig-learning")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    global LANG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(LABELS), default="en",
                    help="language of the in-figure text (default: en)")
    LANG = ap.parse_args().lang

    for fn in (fig_degradation, fig_learning_curve, fig_maps):
        print(fn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
