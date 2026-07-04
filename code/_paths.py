#!/usr/bin/env python3
"""Resolución de rutas por experimento (track análisis), para correr landscapes
separados y reproducibles sin sobrescribir outputs anteriores.

tag="" → experimento por defecto (broad-crime), rutas legacy.
tag="rq2_reporting" → namespace separado:
    data:    data/silver/rq2_reporting/*
    reportes: analysis/rq2_reporting_landscape.md, analysis/rq2_reporting_bibliometrics.md

La viz vive en viz/ (Observable Framework); el legacy D3 en docs/ fue retirado (slr-rj0).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def paths(tag: str = "") -> SimpleNamespace:
    tag = (tag or "").strip()
    if tag:
        data = ROOT / "data" / "silver" / tag
        landscape_md = ROOT / "analysis" / f"{tag}_landscape.md"
        biblio_md = ROOT / "analysis" / f"{tag}_bibliometrics.md"
    else:
        data = ROOT / "data" / "silver" / "analysis"
        landscape_md = ROOT / "analysis" / "field_landscape.md"
        biblio_md = ROOT / "analysis" / "bibliometrics.md"
    return SimpleNamespace(
        tag=tag,
        data=data,
        corpus=data / "field_corpus.json",
        emb=data / "embeddings.npz",
        field_clusters=data / "field_clusters.json",
        refs=data / "refs_full.json",
        citers=data / "citers.json",
        coupling=data / "coupling.json",
        cocitation=data / "cocitation.json",
        citation_net=data / "citation_net.json",
        core_clusters=data / "core_clusters.json",
        landscape_md=landscape_md,
        biblio_md=biblio_md,
    )
