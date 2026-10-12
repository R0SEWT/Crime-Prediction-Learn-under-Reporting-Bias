"""Shared MLflow tracking helpers for Etapa 2 experiments."""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, str]:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "unknown"

    commit = os.environ.get("GIT_COMMIT") or _git("rev-parse", "--short", "HEAD")
    commit_file = root / ".git_commit"
    if commit == "unknown" and commit_file.exists():
        commit = commit_file.read_text(encoding="utf-8").strip() or "unknown"

    dirty_env = os.environ.get("GIT_DIRTY")
    status = _git("status", "--porcelain")
    if dirty_env is not None:
        dirty = dirty_env
    elif status == "unknown":
        dirty = "unknown"
    else:
        dirty = "true" if status else "false"

    return {
        "git_commit": commit,
        "git_dirty": dirty,
    }


def parquet_info(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "rows": 0,
            "cols": 0,
            "sha256": "",
            "columns": [],
        }

    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        rows = parquet.metadata.num_rows
        cols = parquet.metadata.num_columns
    except Exception:
        df = pd.read_parquet(path)
        columns = list(df.columns)
        rows, cols = df.shape

    return {
        "path": str(path),
        "exists": True,
        "rows": int(rows),
        "cols": int(cols),
        "sha256": file_sha256(path),
        "columns": list(columns),
    }


def detect_feature_set(columns: list[str]) -> str:
    groups = []
    checks = [
        ("admin", {"ubigeo", "distrito"}),
        ("population", {"population", "pop_density_km2"}),
        ("meta", {"pop_meta_2020", "pop_meta_density_km2"}),
        ("viirs", {"dnb_mean", "dnb_std"}),
        ("streetview", set()),
        ("osm", {"road_density_km_km2", "intersection_count"}),
        ("proximity", {"dist_police_km", "dist_metro_km", "dist_brt_km"}),
        ("srtm", {"elevation_m", "slope_deg"}),
        ("landcover", {"lc_builtup", "lc_bare", "lc_water"}),
        ("poverty", {"nbi_pct", "poverty_decile"}),
        ("redatam", {"avg_household_size"}),
        ("sentinel2", set()),
    ]
    column_set = set(columns)
    for name, required in checks:
        if name == "streetview":
            present = any(col.startswith("sv_emb_") for col in columns)
        elif name == "sentinel2":
            present = any(col.startswith("s2_") for col in columns)
        else:
            present = bool(required & column_set)
        if present:
            groups.append(name)
    return "+".join(groups) if groups else "unknown"


def dataset_version_from_feature_set(feature_set: str) -> str:
    groups = set(feature_set.split("+"))
    version = "h3base"
    additions = []
    if "meta" in groups:
        additions.append("meta")
    if "sentinel2" in groups:
        additions.append("s2")
    if additions:
        version = f"{version}+{'+'.join(additions)}"
    return version


def dataset_fingerprint(matrix_path: Path, *, graph_path: Path | None = None, target_path: Path | None = None) -> dict:
    matrix = parquet_info(matrix_path)
    graph = parquet_info(graph_path) if graph_path else None
    target = parquet_info(target_path) if target_path else None
    feature_set = detect_feature_set(matrix["columns"])

    params = {
        "dataset_feature_set": feature_set,
        "feature_matrix_rows": matrix["rows"],
        "feature_matrix_cols": matrix["cols"],
        "feature_matrix_sha256": matrix["sha256"],
        "feature_matrix_path": matrix["path"],
    }
    if graph:
        params.update({
            "graph_rows": graph["rows"],
            "graph_cols": graph["cols"],
            "graph_sha256": graph["sha256"],
            "graph_path": graph["path"],
        })
    if target:
        params.update({
            "target_rows": target["rows"],
            "target_cols": target["cols"],
            "target_sha256": target["sha256"],
            "target_path": target["path"],
        })

    return {"feature_set": feature_set, "params": params}


def semantic_run_name(
    *,
    model_version: str,
    model_family: str,
    architecture: str,
    dataset_version: str,
    stage: str,
    date: datetime | None = None,
) -> str:
    run_date = (date or datetime.now()).strftime("%Y%m%d")
    parts = [
        model_version,
        model_family,
        architecture,
        dataset_version,
        stage,
        run_date,
    ]
    return "-".join(_slug(part) for part in parts if part)


def common_tags(
    *,
    root: Path,
    model_family: str,
    model_version: str,
    architecture: str,
    dataset_version: str,
    target_version: str,
    run_stage: str,
) -> dict[str, str]:
    tags = {
        "model_family": model_family,
        "model_version": model_version,
        "architecture": architecture,
        "dataset_version": dataset_version,
        "target_version": target_version,
        "stage": run_stage,
    }
    tags.update(git_metadata(root))
    return tags


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    out = []
    for char in text:
        if char.isalnum():
            out.append(char)
        elif char in {"+", ".", "_", "-"}:
            out.append(char)
        elif char.isspace() or char in {"/", ":", ","}:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug
