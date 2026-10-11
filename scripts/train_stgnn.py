#!/usr/bin/env python3
"""
STGNN — Spatiotemporal GNN para disaggregación espacial de riesgo delictivo.

Arquitectura:
  1. SV projector:    Linear(1536→64) → ReLU  [nodo, 64]
  2. Static fusion:   concat(static_24, sv_proj_64, sv_flag_1) → Linear(89→64) → ReLU
  3. VIIRS temporal:  GRU(input=2, hidden=32, T=5) → [nodo, 32]
  4. GCN concat:      concat(static_emb_64, viirs_emb_32) → GCNConv(96→64) → ReLU
  5. GCN2:            GCNConv(64→64) → ReLU
  6. Heads:           5 × Linear(64→1) → Softplus

Loss:
  L_district  = Σ_d ||Σ_{h∈d} ŷ_h - y_d||² / |d|   (conservación del latente distrital)
  L_prior     = MSE(ŷ, prior) normalizado por categoría (no alejarse del prior poblacional)
  L = L_district + λ * L_prior

Training: años 2019-2022 (train) / 2023 (test). District-stratified.

Usage (local CPU o Lightning AI GPU):
    pip install -r requirements_etapa2.txt
    python3 scripts/train_stgnn.py
    python3 scripts/train_stgnn.py --epochs 200 --hidden 128 --lambda_prior 0.3
    python3 scripts/train_stgnn.py --fast-dev-run   # smoke test (1 batch)

Output:
    data/silver/predictions/stgnn_latente_h3.parquet
    analysis/stgnn_metrics.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiment_tracking import (
        common_tags,
        dataset_fingerprint,
        dataset_version_from_feature_set,
        semantic_run_name,
    )
except ModuleNotFoundError:
    from scripts.experiment_tracking import (
        common_tags,
        dataset_fingerprint,
        dataset_version_from_feature_set,
        semantic_run_name,
    )

ROOT = Path(__file__).resolve().parents[1]
MATRIX_FILE = ROOT / "data/silver/h3_feature_matrix.parquet"
CRIME_FILE = ROOT / "data/silver/crime_latent_surface.parquet"
HYBRID_FILE = ROOT / "data/silver/crime_latent_surface_hybrid.parquet"
GRAPH_FILE = ROOT / "data/silver/h3_graph_edges.parquet"
OUT_DIR = ROOT / "data/silver/predictions"
REPORT_FILE = ROOT / "analysis/stgnn_metrics.md"
MLFLOW_DIR = ROOT / "mlruns"
MLFLOW_DB = ROOT / "mlflow.db"

CRIME_CATS = [
    "robo_hurto_callejero",
    "extorsion",
    "estafa",
    "violencia_familiar_sexual",
    "secuestro",
]
TARGET_COLS = [f"latente_{c}" for c in CRIME_CATS]

STATIC_COLS = [
    "population", "pop_density_km2",
    "pop_meta_2020", "pop_meta_density_km2",
    "elevation_m", "slope_deg", "elevation_std",
    "dist_police_km", "dist_metro_km", "dist_brt_km",
    "lc_builtup", "lc_bare", "lc_water", "lc_trees", "lc_grass",
    "nbi_pct", "poverty_decile", "avg_household_size",
    # OSM
    "road_density_km_km2", "intersection_count",
    "poi_count_education", "poi_count_healthcare", "poi_count_transport",
    "poi_count_food", "poi_count_retail", "poi_count_finance", "poi_count_nightlife",
    # Sentinel-2 composite estático (s2_n_scenes excluido — columna constante)
    "s2_ndvi_median", "s2_ndbi_median", "s2_valid_pixel_ratio",
]
VIIRS_COLS = ["dnb_mean", "dnb_std"]
SV_COLS_PREFIX = "sv_emb_"
SV_DIM = 1536

TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023
ALL_YEARS = sorted(TRAIN_YEARS + [TEST_YEAR])
N_YEARS = len(ALL_YEARS)
YEAR_IDX = {y: i for i, y in enumerate(ALL_YEARS)}
RANDOM_SEED = 42


def default_mlflow_tracking_uri() -> str:
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    if MLFLOW_DB.exists():
        return f"sqlite:///{MLFLOW_DB}"
    return f"file://{MLFLOW_DIR}"


def metric_key(*parts: str) -> str:
    return "_".join(p.replace("/", "_") for p in parts)


def build_mlflow_logger(args: argparse.Namespace):
    if args.no_mlflow:
        return None
    if args.mlflow_tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        from lightning.pytorch.loggers import MLFlowLogger
    except ImportError:
        print("MLflow logger unavailable; continuing without experiment tracking.")
        return None
    run_name = args.run_name or semantic_run_name(
        model_version=args.model_version,
        model_family="stgnn",
        architecture=args.architecture,
        dataset_version=args.dataset_version,
        stage=args.run_stage,
    )
    logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=run_name,
        tracking_uri=args.mlflow_tracking_uri,
    )
    tags = common_tags(
        root=ROOT,
        model_family="stgnn",
        model_version=args.model_version,
        architecture=args.architecture,
        dataset_version=args.dataset_version,
        target_version=args.target_version,
        run_stage=args.run_stage,
    )
    for key, value in tags.items():
        logger.experiment.set_tag(logger.run_id, key, value)
    return logger


def log_stgnn_artifacts(logger, paths: list[Path]) -> None:
    if logger is None:
        return
    for path in paths:
        if path.is_file():
            logger.experiment.log_artifact(logger.run_id, str(path))


def log_stgnn_eval_metrics(logger, rows: list[dict]) -> None:
    if logger is None:
        return
    metrics = {}
    params = {}
    for row in rows:
        cat = row["cat"]
        metrics[metric_key(cat, "mae")] = row["mae"]
        metrics[metric_key(cat, "spearman")] = row["rho"]
        metrics[metric_key(cat, "hs100")] = row["hs100"]
        params[metric_key(cat, "n_test")] = row["n_test"]
    if rows:
        metrics["test_mae_macro"] = float(np.mean([row["mae"] for row in rows]))
        metrics["test_spearman_macro"] = float(np.mean([row["rho"] for row in rows]))
        metrics["test_hs100_macro"] = float(np.mean([row["hs100"] for row in rows]))
        metrics["test_spearman_min"] = float(min(row["rho"] for row in rows))
    if metrics:
        for key, value in metrics.items():
            logger.experiment.log_metric(logger.run_id, key, value)
    if params:
        for key, value in params.items():
            logger.experiment.log_param(logger.run_id, key, value)


# ─── Data preparation ────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading feature matrix...", flush=True)
    df = pd.read_parquet(MATRIX_FILE)
    print(f"  {df.shape}", flush=True)

    # ubigeo is now embedded in the matrix (from h3_admin.parquet bridge)
    if "ubigeo" not in df.columns:
        crime_ub = pd.read_parquet(CRIME_FILE, columns=["h3_index", "ubigeo"]).drop_duplicates("h3_index")
        df = df.merge(crime_ub, on="h3_index", how="left")
    crime = pd.DataFrame()  # no longer needed separately

    print("Loading graph edges...", flush=True)
    if not GRAPH_FILE.exists():
        print(f"  WARNING: {GRAPH_FILE} not found. Run scripts/build_h3_graph.py first.")
        print("  Building edges on the fly (slow)...")
        edges = _build_edges_inline(df["h3_index"].unique().tolist())
    else:
        edges = pd.read_parquet(GRAPH_FILE)
        print(f"  {len(edges):,} directed edges", flush=True)

    return df, crime, edges


def _build_edges_inline(cells: list[str]) -> pd.DataFrame:
    import h3
    from tqdm import tqdm
    cell_set = set(cells)
    rows = []
    for cell in tqdm(cells, desc="Building H3 adjacency"):
        for nb in h3.grid_disk(cell, 1):
            if nb != cell and nb in cell_set:
                rows.append((cell, nb))
    return pd.DataFrame(rows, columns=["src_h3", "dst_h3"])


def impute_median(arr: np.ndarray) -> np.ndarray:
    """Column-wise median imputation for 2D array."""
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = np.isnan(col)
        if mask.any():
            median = np.nanmedian(col)
            col[mask] = median if not np.isnan(median) else 0.0
            arr[:, j] = col
    return arr


def build_tensors(df: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Build all tensors for the STGNN. Returns dict of numpy arrays."""
    import torch

    cells = sorted(df["h3_index"].unique().tolist())
    cell_idx = {c: i for i, c in enumerate(cells)}
    N = len(cells)
    print(f"Nodes: {N}", flush=True)

    # ── Static features [N, F_s] ──
    static_cols_avail = [c for c in STATIC_COLS if c in df.columns]
    df_static = (
        df[df["year"] == TRAIN_YEARS[-1]]
        .set_index("h3_index")[static_cols_avail]
        .reindex(cells)
    )
    x_static = impute_median(df_static.values.astype(np.float32))

    # ── VIIRS features [N, T, 2] ──
    x_viirs = np.zeros((N, N_YEARS, 2), dtype=np.float32)
    for t, yr in enumerate(ALL_YEARS):
        df_yr = df[df["year"] == yr].set_index("h3_index")[VIIRS_COLS].reindex(cells)
        viirs_yr = impute_median(df_yr.values.astype(np.float32))
        x_viirs[:, t, :] = viirs_yr

    # ── SV embeddings [N, SV_DIM] ──
    sv_cols = [c for c in df.columns if c.startswith(SV_COLS_PREFIX)][:SV_DIM]
    sv_available = np.zeros(N, dtype=np.float32)
    x_sv = np.zeros((N, len(sv_cols) if sv_cols else SV_DIM), dtype=np.float32)
    if sv_cols:
        df_sv = (
            df[df["year"] == TRAIN_YEARS[-1]]
            .set_index("h3_index")[sv_cols]
            .reindex(cells)
        )
        has_sv_mask = df_sv.iloc[:, 0].notna().values
        sv_available[has_sv_mask] = 1.0
        sv_vals = df_sv.values.astype(np.float32)
        sv_vals[np.isnan(sv_vals)] = 0.0
        x_sv = sv_vals

    # ── Crime targets [N, T, n_cats] ──
    y_target = np.full((N, N_YEARS, len(CRIME_CATS)), np.nan, dtype=np.float32)
    for t, yr in enumerate(ALL_YEARS):
        df_yr = df[df["year"] == yr].set_index("h3_index")
        for k, col in enumerate(TARGET_COLS):
            if col in df_yr.columns:
                vals = df_yr[col].reindex(cells).values.astype(np.float32)
                y_target[:, t, k] = vals

    # ── District mapping ──
    ubigeo_series = df[df["year"] == TRAIN_YEARS[-1]].set_index("h3_index")["ubigeo"].reindex(cells)
    ubigeos = ubigeo_series.values

    # ── Graph edges ──
    src_list = [cell_idx[s] for s in edges["src_h3"] if s in cell_idx]
    dst_list = [cell_idx[d] for d in edges["dst_h3"] if d in cell_idx]
    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    # ── Train/test split by district ──
    unique_ubigeos = [u for u in pd.Series(ubigeos).dropna().unique() if u != ""]
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_ubigeos)
    n_test = max(1, int(len(unique_ubigeos) * 0.25))
    test_ubigeos = set(unique_ubigeos[:n_test])
    train_mask = np.array([
        (u not in (None, "")) and (u not in test_ubigeos) and not pd.isna(u)
        for u in ubigeos
    ], dtype=bool)
    test_mask = np.array([
        (u not in (None, "")) and (u in test_ubigeos) and not pd.isna(u)
        for u in ubigeos
    ], dtype=bool)

    print(f"Train nodes (with district): {train_mask.sum()}", flush=True)
    print(f"Test nodes  (with district): {test_mask.sum()}", flush=True)

    # ── District totals [n_districts, T, n_cats] for consistency loss ──
    # computed as sum of H3 targets in each district
    district_list = sorted(set(u for u in ubigeos if u and not pd.isna(u)))
    dist_idx = {d: i for i, d in enumerate(district_list)}
    n_dist = len(district_list)
    dist_node_map: list[list[int]] = [[] for _ in range(n_dist)]
    for i, u in enumerate(ubigeos):
        if u and not pd.isna(u) and u in dist_idx:
            dist_node_map[dist_idx[u]].append(i)

    # For train districts: sum of y_target over train years
    train_dist_idx = [
        dist_idx[u] for u in unique_ubigeos if u not in test_ubigeos and u in dist_idx
    ]
    train_year_indices = [YEAR_IDX[y] for y in TRAIN_YEARS]
    train_targets = y_target[train_mask][:, train_year_indices, :].reshape(-1, len(CRIME_CATS))
    cat_scale = np.nanmean(np.abs(train_targets), axis=0).astype(np.float32)
    cat_scale = np.where(np.isfinite(cat_scale) & (cat_scale > 0), cat_scale, 1.0).astype(np.float32)

    print(f"Districts (train): {len(train_dist_idx)}  Total: {n_dist}", flush=True)
    print(
        "Target scales: "
        + ", ".join(f"{cat}={scale:.3g}" for cat, scale in zip(CRIME_CATS, cat_scale)),
        flush=True,
    )

    return {
        "cells": cells,
        "cell_idx": cell_idx,
        "N": N,
        "x_static": x_static,
        "x_viirs": x_viirs,
        "x_sv": x_sv,
        "sv_available": sv_available,
        "y_target": y_target,
        "edge_index": edge_index,
        "ubigeos": ubigeos,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "dist_node_map": dist_node_map,
        "train_dist_idx": train_dist_idx,
        "dist_idx": dist_idx,
        "n_dist": n_dist,
        "cat_scale": cat_scale,
        "train_year_indices": train_year_indices,
        "test_year_index": YEAR_IDX[TEST_YEAR],
    }


# ─── Model ───────────────────────────────────────────────────────────────────

def build_model_and_module(data: dict, args: argparse.Namespace):
    import torch
    import torch.nn as nn
    import lightning as L
    from torch_geometric.nn import GCNConv

    n_static = data["x_static"].shape[1]
    n_sv_input = data["x_sv"].shape[1]
    hidden = args.hidden
    n_cats = len(CRIME_CATS)
    fusion = getattr(args, "fusion", "concat")

    class CrimeSTGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion = fusion
            if fusion == "attention":
                # Cada modalidad -> un token de dimension `hidden`; la fusion
                # es atencion multi-cabeza sobre los tokens (SV, estatico, temporal)
                # en lugar de la concatenacion del baseline.
                self.sv_proj = nn.Sequential(
                    nn.Linear(n_sv_input, hidden), nn.ReLU(), nn.Dropout(0.2)
                )
                self.static_enc = nn.Sequential(
                    nn.Linear(n_static + 1, hidden), nn.ReLU(), nn.Dropout(0.2)
                )
                viirs_hidden = hidden
                self.viirs_gru = nn.GRU(
                    input_size=2, hidden_size=viirs_hidden,
                    num_layers=1, batch_first=False,
                )
                self.modal_attn = nn.MultiheadAttention(
                    embed_dim=hidden, num_heads=4, dropout=0.2, batch_first=True
                )
                self.fusion_norm = nn.LayerNorm(hidden)
                gcn_in = hidden
            else:
                # SV projector
                sv_proj_dim = min(64, hidden)
                self.sv_proj = nn.Sequential(
                    nn.Linear(n_sv_input, sv_proj_dim), nn.ReLU(), nn.Dropout(0.2)
                )
                # Static + SV fusion: static + sv_proj + sv_flag
                self.static_enc = nn.Sequential(
                    nn.Linear(n_static + sv_proj_dim + 1, hidden), nn.ReLU(), nn.Dropout(0.2)
                )
                # Temporal VIIRS encoder: GRU over T time steps
                viirs_hidden = hidden // 2
                self.viirs_gru = nn.GRU(
                    input_size=2, hidden_size=viirs_hidden,
                    num_layers=1, batch_first=False,
                )
                # GCN layers: concat(static_emb, viirs_emb) as input
                gcn_in = hidden + viirs_hidden
            self.gcn1 = GCNConv(gcn_in, hidden)
            self.gcn2 = GCNConv(hidden, hidden)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(0.2)
            # Independent heads reduce cross-category interference for rare targets.
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, hidden // 2), nn.ReLU(),
                    nn.Linear(hidden // 2, 1), nn.Softplus(),
                )
                for _ in range(n_cats)
            ])

        def forward(self, x_static, x_sv, sv_flag, x_viirs, edge_index):
            # x_static: [N, F_s]
            # x_sv: [N, sv_dim]
            # sv_flag: [N, 1]
            # x_viirs: [N, T, 2]
            # edge_index: [2, E]
            N = x_static.shape[0]

            if self.fusion == "attention":
                # Un token por modalidad, todos de dimension `hidden`
                sv_token = self.sv_proj(x_sv)  # [N, hidden]
                static_token = self.static_enc(
                    torch.cat([x_static, sv_flag], dim=-1)
                )  # [N, hidden]
                viirs_in = x_viirs.permute(1, 0, 2)  # [T, N, 2]
                viirs_out, _ = self.viirs_gru(viirs_in)  # [T, N, hidden]
                viirs_token = viirs_out.mean(dim=0)  # [N, hidden]

                # Atencion cross-modal: cada modalidad atiende a las demas.
                tokens = torch.stack(
                    [sv_token, static_token, viirs_token], dim=1
                )  # [N, 3, hidden]
                attn_out, _ = self.modal_attn(tokens, tokens, tokens)  # [N, 3, hidden]
                # Pool de tokens fusionados + conexion residual, normalizado
                h = self.fusion_norm(attn_out.mean(dim=1) + tokens.mean(dim=1))  # [N, hidden]
            else:
                # SV projection
                sv_emb = self.sv_proj(x_sv)  # [N, sv_proj_dim]
                # Static + SV fusion
                h_static = self.static_enc(
                    torch.cat([x_static, sv_emb, sv_flag], dim=-1)
                )  # [N, hidden]
                # VIIRS GRU: input [T, N, 2], output [T, N, viirs_hidden]
                viirs_in = x_viirs.permute(1, 0, 2)  # [T, N, 2]
                viirs_out, _ = self.viirs_gru(viirs_in)  # [T, N, viirs_hidden]
                h_viirs = viirs_out.mean(dim=0)  # [N, viirs_hidden]
                # GCN on combined embedding
                h = torch.cat([h_static, h_viirs], dim=-1)  # [N, hidden + viirs_hidden]

            h = self.act(self.gcn1(h, edge_index))
            h = self.drop(h)
            h = self.act(self.gcn2(h, edge_index))  # [N, hidden]

            # Output: [N, n_cats]
            return torch.cat([head(h) for head in self.heads], dim=-1)

    class CrimeLightningModule(L.LightningModule):
        def __init__(self, model, data, args):
            super().__init__()
            self.model = model
            self.args = args
            self.lambda_prior = args.lambda_prior
            self.lambda_district = args.lambda_district

            # Register buffers (GPU-transferable)
            import torch as th
            self.register_buffer("x_static", th.tensor(data["x_static"]))
            self.register_buffer("x_sv", th.tensor(data["x_sv"]))
            self.register_buffer("sv_flag", th.tensor(data["sv_available"]).unsqueeze(-1))
            self.register_buffer("x_viirs", th.tensor(data["x_viirs"]))
            self.register_buffer("edge_index", th.tensor(data["edge_index"]))
            self.register_buffer("y_target", th.tensor(data["y_target"]))
            self.register_buffer(
                "cat_scale",
                th.tensor(data["cat_scale"]).clamp_min(args.loss_scale_floor),
            )
            # masks
            self.register_buffer("train_mask", th.tensor(data["train_mask"]))
            self.register_buffer("test_mask", th.tensor(data["test_mask"]))

            # District consistency structure
            self.dist_node_map = data["dist_node_map"]
            self.train_dist_idx = data["train_dist_idx"]
            self.train_year_indices = data["train_year_indices"]
            self.test_year_index = data["test_year_index"]
            self.n_cats = n_cats

        def forward(self):
            return self.model(
                self.x_static, self.x_sv, self.sv_flag,
                self.x_viirs, self.edge_index,
            )  # [N, n_cats]

        def _loss(self, pred, split="train"):
            import torch as th
            # pred: [N, n_cats]
            mask = self.train_mask if split == "train" else self.test_mask
            year_indices = self.train_year_indices if split == "train" else [self.test_year_index]

            total_loss = th.tensor(0.0, device=self.device)
            n_terms = 0

            for t in year_indices:
                y_t = self.y_target[:, t, :]  # [N, n_cats]
                valid = mask & ~th.isnan(y_t[:, 0])  # nodes with this-year target

                if valid.sum() == 0:
                    continue

                pred_valid = pred[valid]
                y_valid = y_t[valid]

                # Prior regularization: per-category MSE, skip NaN cats
                l_prior = th.tensor(0.0, device=self.device)
                n_prior = 0
                for k in range(self.n_cats):
                    y_k = y_valid[:, k]
                    ok_k = ~th.isnan(y_k)
                    if ok_k.sum() > 0:
                        err = (pred_valid[ok_k, k] - y_k[ok_k]) / self.cat_scale[k]
                        l_prior = l_prior + th.mean(err ** 2)
                        n_prior += 1
                if n_prior > 0:
                    l_prior = l_prior / n_prior

                # District consistency: sum over H3 in district = district total
                l_dist = th.tensor(0.0, device=self.device)
                n_d = 0
                dist_indices = self.train_dist_idx if split == "train" else [
                    i for i in range(len(self.dist_node_map))
                    if any(self.test_mask[j] for j in self.dist_node_map[i])
                ]
                for di in dist_indices:
                    node_list = self.dist_node_map[di]
                    if not node_list:
                        continue
                    node_t = th.tensor(node_list, device=self.device)
                    y_nodes = y_t[node_t]
                    # per-category district consistency, skip NaN cats
                    l_d_cat = th.tensor(0.0, device=self.device)
                    n_d_cat = 0
                    for k in range(self.n_cats):
                        y_nk = y_nodes[:, k]
                        ok = ~th.isnan(y_nk)
                        if ok.sum() == 0:
                            continue
                        y_dist_k = y_nk[ok].sum()
                        pred_dist_k = pred[node_t][ok, k].sum()
                        denom = self.cat_scale[k] * (ok.sum().float() + 1.0).sqrt()
                        l_d_cat = l_d_cat + ((pred_dist_k - y_dist_k) / denom) ** 2
                        n_d_cat += 1
                    if n_d_cat > 0:
                        l_dist = l_dist + l_d_cat / n_d_cat
                        n_d += 1

                if n_d > 0:
                    l_dist = l_dist / n_d

                loss_t = self.lambda_district * l_dist + self.lambda_prior * l_prior
                total_loss = total_loss + loss_t
                n_terms += 1

            return total_loss / max(n_terms, 1) if n_terms > 0 else total_loss

        def training_step(self, batch, batch_idx):
            pred = self()
            loss = self._loss(pred, split="train")
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            import torch as th
            pred = self()
            loss = self._loss(pred, split="test")
            self.log("val_loss", loss, prog_bar=True)

            # Spearman-like metric: rank correlation on test nodes, per category.
            t = self.test_year_index
            y_t = self.y_target[:, t, :]
            for k, cat in enumerate(CRIME_CATS):
                valid = self.test_mask & ~th.isnan(y_t[:, k])
                if valid.sum() <= 5:
                    continue
                y_vals = y_t[valid, k].cpu().numpy()
                p_vals = pred[valid, k].detach().cpu().numpy()
                from scipy.stats import spearmanr
                rho, _ = spearmanr(p_vals, y_vals)
                rho = float(rho) if not np.isnan(rho) else 0.0
                self.log(f"val_spearman_{cat}", rho, prog_bar=(k == 0))
                if k == 0:
                    self.log("val_spearman", rho, prog_bar=True)
            return loss

        def configure_optimizers(self):
            optimizer = __import__("torch").optim.AdamW(
                self.parameters(), lr=self.args.lr, weight_decay=1e-4
            )
            scheduler = __import__("torch").optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs, eta_min=1e-5
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    model = CrimeSTGNN()
    module = CrimeLightningModule(model, data, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)
    return module


# ─── Training ─────────────────────────────────────────────────────────────────

def train(data: dict, args: argparse.Namespace):
    import torch
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

    # el split distrital queda SIEMPRE fijo (RANDOM_SEED en build_tensors);
    # --seed solo varía la inicialización/entrenamiento para estudios multi-seed
    L.seed_everything(getattr(args, "seed", RANDOM_SEED), workers=True)

    # Dummy dataset — the model uses global tensors (full-graph, no mini-batching)
    dummy = TensorDataset(torch.zeros(1))
    train_loader = DataLoader(dummy, batch_size=1)
    val_loader = DataLoader(dummy, batch_size=1)

    module = build_model_and_module(data, args)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=30, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(OUT_DIR / "checkpoints"),
            filename="stgnn-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss", mode="min", save_top_k=1,
        ),
    ]
    logger = build_mlflow_logger(args)
    if logger is not None:
        logger.log_hyperparams({
            "model": "stgnn",
            "model_version": args.model_version,
            "architecture": args.architecture,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "lambda_prior": args.lambda_prior,
            "lambda_district": args.lambda_district,
            "loss_scale_floor": args.loss_scale_floor,
            "train_years": ",".join(map(str, TRAIN_YEARS)),
            "test_year": TEST_YEAR,
            "random_seed": RANDOM_SEED,
            "n_nodes": data["N"],
            "n_edges": int(data["edge_index"].shape[1]),
            "n_train_nodes": int(data["train_mask"].sum()),
            "n_test_nodes": int(data["test_mask"].sum()),
            "split": "district-stratified",
            "test_district_frac": 0.25,
            "dataset_version": args.dataset_version,
            "target_version": args.target_version,
            **getattr(args, "dataset_tracking_params", {}),
            **{
                f"target_scale_{cat}": float(scale)
                for cat, scale in zip(CRIME_CATS, data["cat_scale"])
            },
        })

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        fast_dev_run=args.fast_dev_run,
        check_val_every_n_epoch=5,
        gradient_clip_val=1.0,
    )

    print(f"\nTraining on {trainer.accelerator.__class__.__name__}...", flush=True)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return module, trainer, logger


# ─── Predict & save ──────────────────────────────────────────────────────────

def restore_best_checkpoint(module, trainer) -> None:
    best_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    if not best_path:
        return
    import torch
    checkpoint = torch.load(best_path, map_location=module.device, weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])
    print(f"Restored best checkpoint: {best_path}")


def save_predictions(module, data: dict, args: argparse.Namespace) -> pd.DataFrame:
    import torch
    module.eval()
    with torch.no_grad():
        pred = module().cpu().numpy()  # [N, n_cats]

    cells = data["cells"]
    rows = []
    for year in ALL_YEARS:
        for i, cell in enumerate(cells):
            row = {"h3_index": cell, "year": year}
            for k, cat in enumerate(CRIME_CATS):
                row[f"pred_{cat}"] = float(pred[i, k])
                t = YEAR_IDX[year]
                row[f"latente_{cat}"] = float(data["y_target"][i, t, k])
            rows.append(row)

    df_pred = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_hybrid" if getattr(args, "target", "latente") == "hybrid" else ""
    suffix += "_attention" if getattr(args, "fusion", "concat") == "attention" else ""
    out_path = OUT_DIR / f"stgnn_latente_h3{suffix}.parquet"
    df_pred.to_parquet(out_path, index=False)
    print(f"Predictions saved: {out_path}  ({len(df_pred):,} rows)")
    return df_pred


def write_report(module, df_pred: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    from scipy.stats import spearmanr

    rows = []
    metric_rows = []
    for cat in CRIME_CATS:
        pred_col = f"pred_{cat}"
        true_col = f"latente_{cat}"
        df_test = df_pred[(df_pred["year"] == TEST_YEAR) & df_pred[true_col].notna()].copy()
        if len(df_test) < 5:
            continue
        y_true = df_test[true_col].values
        y_pred = df_test[pred_col].values
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rho, _ = spearmanr(y_pred, y_true)
        rho = float(rho) if not np.isnan(rho) else 0.0
        # Hotspot@100
        k = min(100, len(y_true) // 5)
        top_true = set(np.argsort(y_true)[-k:])
        top_pred = set(np.argsort(y_pred)[-k:])
        hs = len(top_true & top_pred) / k
        rows.append({
            "cat": cat, "n_test": len(df_test),
            "MAE": round(mae, 3), "ρ": round(rho, 3), "HS@100": f"{hs:.1%}",
        })
        metric_rows.append({
            "cat": cat, "n_test": len(df_test),
            "mae": mae, "rho": rho, "hs100": hs,
        })

    table = pd.DataFrame(rows).to_markdown(index=False) if rows else "_no test predictions_"
    fusion_mode = getattr(args, "fusion", "concat")
    if fusion_mode == "attention":
        arch_desc = (
            "SV(→hidden) · Static+flag(→hidden) · VIIRS(2)×5→GRU(→hidden)   [3 tokens de modalidad]\n"
            "MultiheadAttention(3 tokens, heads=4) → mean + residual → LayerNorm   [fusion cross-modal]\n"
            "→ GCNConv(hidden→hidden) → GCNConv(hidden→hidden)                     [espacial]\n"
            "5 × (Linear(hidden→hidden/2→1) → Softplus)                            [heads independientes]"
        )
    else:
        arch_desc = (
            "SV(1536→64) + Static(24) + sv_flag → Linear(89→64)       [estático por nodo]\n"
            "VIIRS(2) × 5 años → GRU(hidden=32)                       [temporal]\n"
            "concat(64, 32) → GCNConv(96→64) → GCNConv(64→64)         [espacial]\n"
            "5 × (Linear(64→32→1) → Softplus)                         [heads independientes]"
        )
    target_mode = getattr(args, "target", "latente")
    if target_mode == "hybrid":
        caveat = (
            "> Target HÍBRIDO (slr-us4): patrón geocodificado suavizado × escala\n"
            "> de-sesgada distrital. Estos ρ son contra ese target (test 2023, cuyo\n"
            "> patrón comparte fuente con el oráculo) — el resultado defendible sigue\n"
            "> siendo la eval directa vs oráculo (`eval_fusion_oracle.py`)."
        )
    else:
        caveat = (
            "> ⚠️ **LECTURA CIRCULAR — NO usar como resultado de tesis.** Estos ρ se calculan\n"
            "> contra el target latente redistribuido (`latente_*`, prior de redistribución\n"
            "> distrital), no contra el oráculo geocodificado. Por construcción el STGNN\n"
            "> aprende ese prior, así que ρ alto mide \"reproduce el prior\", **no** \"aprende\n"
            "> riesgo H3 real\". Resultado honesto: `train_stgnn_real.py` vs oráculo\n"
            "> (`ch5-resultados.typ` §Q1)."
        )
    content = f"""# STGNN — Métricas (Etapa 2)

{caveat}

**Target:** {target_mode}  **Fusion:** {fusion_mode}  **Epochs:** {args.epochs}  **Hidden:** {args.hidden}  **λ_prior:** {args.lambda_prior}  **λ_district:** {args.lambda_district}
**Architecture:** independent per-category heads
**Loss scale floor:** {args.loss_scale_floor}
**Train:** {TRAIN_YEARS}  **Test:** {TEST_YEAR}

## Resultados por categoría

{table}

## Arquitectura

```
{arch_desc}
```

## Loss

```
L = {args.lambda_district} × L_district + {args.lambda_prior} × L_prior
L_district = mean_cat ((Σ_h∈d ŷ_h,c − y_d,c) / (scale_c × sqrt(|d|)))²
L_prior    = mean_cat MSE((ŷ_c − prior_c) / scale_c)
```
"""
    if target_mode == "hybrid":
        report_path = REPORT_FILE.parent / f"stgnn_hybrid_{fusion_mode}_metrics.md"
    elif fusion_mode == "attention":
        report_path = REPORT_FILE.parent / "stgnn_fusion_metrics.md"
    else:
        report_path = REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    print(f"Report: {report_path}")
    return metric_rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="seed de entrenamiento (el split distrital no cambia)")
    p.add_argument("--target", choices=["latente", "hybrid"], default="latente",
                   help="target H3: latente (patrón poblacional, circular) o hybrid "
                        "(patrón geocodificado × escala de-sesgada, slr-us4)")
    p.add_argument("--fusion", choices=["concat", "attention"], default="concat",
                   help="Fusion cross-modal: 'concat' (baseline) o 'attention' "
                        "(atencion multi-cabeza sobre tokens de modalidad SV/estatico/temporal)")
    p.add_argument("--lambda-prior", dest="lambda_prior", type=float, default=0.5)
    p.add_argument("--lambda-district", dest="lambda_district", type=float, default=1.0,
                   help="Peso de L_district; 1.0 = comportamiento canónico, 0 = ablación (EXP-2)")
    p.add_argument("--loss-scale-floor", type=float, default=1.0,
                   help="Minimum per-category target scale used to normalize loss")
    p.add_argument("--fast-dev-run", dest="fast_dev_run", action="store_true",
                   help="Lightning fast_dev_run — smoke test only")
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    p.add_argument("--mlflow-tracking-uri", default=default_mlflow_tracking_uri())
    p.add_argument("--experiment-name", default="infelix-etapa2-stgnn")
    p.add_argument("--run-name", default=None)
    p.add_argument("--model-version", default="stgnn-v003")
    p.add_argument("--architecture", default="rareheads-scaledloss-bestckpt")
    p.add_argument("--dataset-version", default="auto")
    p.add_argument("--target-version", default="latent-v1")
    p.add_argument("--run-stage", default="manual")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Config: epochs={args.epochs} lr={args.lr} hidden={args.hidden} "
          f"lambda_prior={args.lambda_prior}", flush=True)

    # Verify deps early
    try:
        import torch
        import lightning
        import torch_geometric
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("Run: pip install -r requirements_etapa2.txt")
        return 1

    df, _, edges = load_data()
    target_file = CRIME_FILE
    if args.target == "hybrid":
        # sobreescribir latente_* del matrix con la superficie híbrida
        # (patrón geocodificado × escala de-sesgada, slr-us4)
        target_file = HYBRID_FILE
        hyb = pd.read_parquet(HYBRID_FILE, columns=["h3_index", "year", "crime_cat", "latente"])
        hyb_wide = (hyb.pivot_table(index=["h3_index", "year"], columns="crime_cat",
                                    values="latente", aggfunc="sum")
                    .rename(columns=lambda c: f"latente_{c}").reset_index())
        df = df.drop(columns=[c for c in TARGET_COLS if c in df.columns])
        df = df.merge(hyb_wide, on=["h3_index", "year"], how="left")
        print(f"Target HÍBRIDO cargado: {HYBRID_FILE.name} "
              f"({int(df[TARGET_COLS[0]].notna().sum()):,} filas con target)", flush=True)
    data = build_tensors(df, edges)
    dataset_info = dataset_fingerprint(MATRIX_FILE, graph_path=GRAPH_FILE, target_path=target_file)
    if args.dataset_version == "auto":
        args.dataset_version = dataset_version_from_feature_set(dataset_info["feature_set"])
    args.dataset_tracking_params = dataset_info["params"]

    module, trainer, logger = train(data, args)

    if not args.fast_dev_run:
        restore_best_checkpoint(module, trainer)
        df_pred = save_predictions(module, data, args)
        metrics = write_report(module, df_pred, args)
        log_stgnn_eval_metrics(logger, metrics)
        suffix = ("_hybrid" if args.target == "hybrid" else "") + (
            "_attention" if args.fusion == "attention" else "")
        log_stgnn_artifacts(
            logger,
            [
                REPORT_FILE,
                OUT_DIR / f"stgnn_latente_h3{suffix}.parquet",
                Path(trainer.checkpoint_callback.best_model_path)
                if trainer.checkpoint_callback.best_model_path else Path(),
            ],
        )

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
