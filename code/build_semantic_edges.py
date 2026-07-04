#!/usr/bin/env python3
"""Edges semánticos del curso por kNN coseno sobre embeddings SPECTER2 locales (slr-dhl).

Lee course_embeddings.npz (vectores L2-normalizados → coseno = producto punto), toma los
top-k vecinos de cada paper, los une simétricamente y mapea paperId→key del stub (los
nodos del grafo del curso usan `key` como id). Solo numpy; no llama a S2 ni a torch.

Uso:
    python3 scripts/analysis/build_semantic_edges.py --k 6 [--min-sim 0.0] [--tag ""]

Salida: data/silver/analysis/course_semantic.json
    {"meta": {...}, "edges": [{"source": key, "target": key, "weight": cos}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import paths  # noqa: E402
import s2_client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="kNN coseno → edges semánticos del curso")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=6, help="vecinos por nodo")
    ap.add_argument("--min-sim", type=float, default=0.0, help="poda por similitud mínima")
    args = ap.parse_args()

    P = paths(args.tag)
    emb_path = P.data / "course_embeddings.npz"
    meta_path = P.data / "course_embeddings_meta.json"
    out_path = P.data / "course_semantic.json"

    if not emb_path.exists():
        sys.exit(f"Falta {emb_path}. Corre embed_course_specter2.py primero.")

    z = np.load(emb_path, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    vecs = z["vecs"].astype(np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)  # asegura L2

    # paperId → key (vía field_corpus.doi + doi_to_key del stub)
    corpus = json.loads(P.corpus.read_text(encoding="utf-8"))
    norm = s2_client.normalize_doi
    doi2key = s2_client.doi_to_key()
    pid2key = {}
    for p in corpus["papers"]:
        if p.get("paperId"):
            pid2key[p["paperId"]] = doi2key.get(norm(p.get("doi") or ""))

    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)  # excluir auto-similitud

    k = max(1, min(args.k, len(ids) - 1))
    seen: set[tuple[str, str]] = set()
    edges = []
    skipped_nokey = 0
    for i, pid in enumerate(ids):
        ka = pid2key.get(pid)
        if not ka:
            skipped_nokey += 1
            continue
        nbr = np.argsort(sim[i])[::-1][:k]
        for j in nbr:
            w = float(sim[i, j])
            if w < args.min_sim:
                continue
            kb = pid2key.get(ids[j])
            if not kb or kb == ka:
                continue
            pair = tuple(sorted((ka, kb)))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({"source": pair[0], "target": pair[1], "weight": round(w, 4)})

    edges.sort(key=lambda e: e["weight"], reverse=True)

    emb_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    out = {
        "meta": {
            "method": "specter2_cosine_knn",
            "k": k,
            "min_sim": args.min_sim,
            "n_nodes_embedded": len(ids),
            "n_nodes_with_key": len(ids) - skipped_nokey,
            "n_edges": len(edges),
            "coverage": emb_meta.get("coverage"),
            "embeddings_meta": "course_embeddings_meta.json",
        },
        "edges": edges,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {len(edges)} edges semánticos (k={k}) → {out_path}")
    print(f"  nodos embebidos: {len(ids)}; con key de stub: {len(ids) - skipped_nokey}")
    if edges:
        print(f"  rango de peso: {edges[-1]['weight']:.3f}–{edges[0]['weight']:.3f}")


if __name__ == "__main__":
    main()
