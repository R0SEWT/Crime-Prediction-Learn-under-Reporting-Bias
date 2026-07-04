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
                            data/silver/predictions/geocoding_degradation.csv
                            (report: analysis/etapa2_geocoding_degradation.md).
fig-simbig-learning.pdf     Local-vs-transfer paired deltas with 95% CIs from
                            analysis/etapa2_learning_curve.md
                            (exp_learning_curve.py, B=2000).
"""

from __future__ import annotations

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
DEGRADATION_CSV = ROOT / "data/silver/predictions/geocoding_degradation.csv"
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
        (gdf_lima, "Lima — geocoded oracle 2023", "Lima — tabular ceiling (M4c)"),
        (gdf_aqp, "Arequipa — own oracle 2023", "Arequipa — Lima-trained transfer"),
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
        cbar.set_label("complaints, log(1+n)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig-simbig-maps.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_degradation() -> Path:
    df = pd.read_csv(DEGRADATION_CSV)
    agg = df.groupby("rate").agg(
        medible=("medible", "mean"), medible_sd=("medible", "std"),
        real=("real", "mean"), real_sd=("real", "std"),
        persist=("persist", "mean"), persist_sd=("persist", "std"),
    ).reset_index().sort_values("rate")
    x = agg["rate"] * 100

    mpl.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(4.9, 3.2))
    # the True/Measurable curves converge at 83.4% by construction, so their
    # end labels need opposite vertical offsets to avoid collision
    series = [
        ("real", "real_sd", "True skill (vs full oracle)", C_BLUE, "-", "o", 8),
        ("medible", "medible_sd", "Measurable skill (vs degraded oracle)", C_VERMILLION, "--", "s", -9),
        ("persist", "persist_sd", "Persistence baseline", C_GREEN, ":", "^", 0),
    ]
    for col, sd, label, color, ls, marker, dy in series:
        ax.plot(x, agg[col], ls, color=color, marker=marker, markersize=4,
                linewidth=1.6, label=label)
        ax.fill_between(x, agg[col] - agg[sd].fillna(0), agg[col] + agg[sd].fillna(0),
                        color=color, alpha=0.15, linewidth=0)
        ax.annotate(label.split(" (")[0], (x.iloc[-1], agg[col].iloc[-1]),
                    xytext=(5, dy), textcoords="offset points",
                    fontsize=7, color=color, va="center")

    # city markers labelled above the axes: outside the data area, so they
    # can never collide with the legend or the series labels
    for rate, name in ((83.4, "Lima"), (25.0, "Trujillo-like")):
        ax.axvline(rate, color=C_GRID, linewidth=0.8, zorder=0)
        ax.annotate(name, (rate, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=6.5, color=C_TEXT)

    ax.set_xlabel("Geocoding rate (%)")
    ax.set_ylabel("Intra-district Spearman ρ (macro)")
    ax.set_xlim(5, 112)
    ax.grid(axis="y", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig-simbig-degradation.pdf"
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

    mpl.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.axhline(0.0, color=C_TEXT, linewidth=1.0)
    ax.annotate("transfer parity (Lima-trained, no local data)", (0.62, 0.0),
                xytext=(0, -10), textcoords="offset points", fontsize=6.5,
                color=C_TEXT, va="top")
    for xi, d, l, h, s in zip(years, delta, lo, hi, sig):
        ax.errorbar(xi, d, yerr=[[d - l], [h - d]], fmt="o", markersize=6,
                    color=C_BLUE, markerfacecolor=C_BLUE if s else "white",
                    markeredgecolor=C_BLUE, capsize=3, linewidth=1.4)
        ax.annotate(f"{d:+.3f}" + ("" if s else " (n.s.)"), (xi, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=7, color=C_TEXT)

    ax.set_xticks(years)
    ax.set_xticklabels([f"{y} yr" for y in years])
    ax.set_xlabel("Local Arequipa training history")
    ax.set_ylabel("Δ macro ρ (local − transfer), 95% CI")
    ax.set_xlim(0.5, 4.5)
    ax.grid(axis="y", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig-simbig-learning.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    for fn in (fig_degradation, fig_learning_curve, fig_maps):
        print(fn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
