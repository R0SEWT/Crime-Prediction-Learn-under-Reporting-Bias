#!/usr/bin/env python3
"""
Supplementary S1 — Clustering robustness check.

Reproduces the k-sensitivity analysis reported in §4.4 of the paper:
"Deep Learning and Spatiotemporal Modelling for Urban Crime Prediction:
A Systematic Review of Architectures, Heterogeneous Data Fusion, and
the Dark Figure Gap (2023–mid 2026)"

Vilchez Marin, R.S. · Velasquez Borasino, C.A. — UPC, 2026.

Usage:
    pip install numpy scipy
    python cluster_robustness.py

Input files (from dois/):
    reporting_landscape_corpus.tsv  — 1,184 papers with SPECTER2 embeddings
    slr_corpus_40papers.tsv         — 40 SLR corpus papers

Embeddings are NOT distributed in this repo (768-dim × 1,184 vectors ≈ 3.4 MB
binary). To reproduce from scratch, obtain a Semantic Scholar API key and run:
    python fetch_embeddings.py --corpus reporting_landscape_corpus.tsv

Expected output:
    k=10: 3 papers (0.25%)  ← reported in paper
    k=12: 3 papers (0.25%)
    k=15: 3 papers (0.25%)
    k=7:  20 papers (1.69%) ← coarse granularity artefact
"""

from __future__ import annotations
import csv
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    from scipy.cluster.vq import kmeans2
except ImportError:
    print("ERROR: pip install numpy scipy")
    sys.exit(1)

HERE = Path(__file__).parent.parent

REPORTING_TERMS = {
    "reporting", "report", "unreported", "under-reporting", "underreport",
    "dark", "figure", "victimization", "victimisation", "survey",
    "trust", "propensity", "willingness", "non-reporting", "police",
}

STOP = set(
    "the a an of and or for to in on with under via using based by from is are "
    "be we our this that these those study studies paper approach method methods "
    "model models results using use used new novel propose proposed framework "
    "analysis data prediction predicting crime crimes between toward towards "
    "into over more most can which their its also has have had not but was were "
    "will been being such than then when where while each other".split()
)


def load_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tokenize(txt: str) -> list[str]:
    txt = (txt or "").lower()
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\\[a-z]+", " ", txt)
    toks = re.findall(r"[a-z][a-z\-]{2,}", txt)
    return [t for t in toks if t not in STOP]


def top_terms(member_indices: list[int], all_docs: list[list[str]], n: int = 6) -> list[str]:
    df: Counter = Counter()
    for doc in all_docs:
        for t in set(doc):
            df[t] += 1
    N = len(all_docs)
    tf: Counter = Counter()
    for i in member_indices:
        for t in all_docs[i]:
            tf[t] += 1
    scored = {t: c * np.log(N / (1 + df[t])) for t, c in tf.items() if df[t] > 1}
    return [t for t, _ in sorted(scored.items(), key=lambda x: -x[1])[:n]]


def is_reporting_cluster(terms: list[str]) -> bool:
    return any(t in REPORTING_TERMS for t in terms)


def run(embeddings_path: Path, corpus_tsv: Path, slr_tsv: Path) -> None:
    # Load embeddings
    d = np.load(embeddings_path, allow_pickle=True)
    ids: list[str] = list(d["ids"])
    vecs: np.ndarray = d["vecs"].astype(np.float32)

    # Load corpus metadata
    corpus = {row["s2_paper_id"]: row for row in load_tsv(corpus_tsv)}

    # SLR paper IDs (match on s2_paper_id if available)
    slr_rows = load_tsv(slr_tsv)
    # SLR papers in landscape = those marked course_selected in corpus
    slr_ids = {pid for pid in ids if corpus.get(pid, {}).get("is_slr_corpus") == "yes"}
    # Fallback: derive from field_corpus course_selected flag
    print(f"Reporting landscape: {len(ids)} papers with embeddings")
    print(f"SLR papers found in landscape: {len(slr_ids)}\n")

    # Normalise L2
    X = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    all_docs = [tokenize(f"{corpus.get(pid, {}).get('title', '')}") for pid in ids]

    print(f"{'k':>4}  {'Reporting clusters':>20}  {'SLR in reporting':>18}  {'%':>6}")
    print("-" * 60)

    for k in [7, 10, 12, 15]:
        np.random.seed(42)
        _, labels = kmeans2(X, k, minit="++", seed=42)
        labels = np.array(labels)

        cluster_members: dict[int, list[int]] = {c: [] for c in range(k)}
        for idx in range(len(ids)):
            cluster_members[int(labels[idx])].append(idx)

        reporting_clusters = set()
        for c, members in cluster_members.items():
            if members and is_reporting_cluster(top_terms(members, all_docs)):
                reporting_clusters.add(c)

        slr_in_reporting = sum(
            1 for idx, pid in enumerate(ids)
            if pid in slr_ids and int(labels[idx]) in reporting_clusters
        )
        pct = 100 * slr_in_reporting / len(ids)
        print(f"{k:>4}  {str(sorted(reporting_clusters)):>20}  {slr_in_reporting:>18}  {pct:>6.2f}%")

    print()
    print("Expected: k∈{10,12,15} → 3 papers (0.25%) each; k=7 inflated by coarse granularity.")


if __name__ == "__main__":
    emb = HERE / "dois" / "reporting_embeddings.npz"
    corpus_tsv = HERE / "dois" / "reporting_landscape_corpus.tsv"
    slr_tsv = HERE / "dois" / "slr_corpus_40papers.tsv"

    if not emb.exists():
        print(f"Embeddings not found at {emb}")
        print("Run: python fetch_embeddings.py --corpus dois/reporting_landscape_corpus.tsv")
        sys.exit(1)

    run(emb, corpus_tsv, slr_tsv)
