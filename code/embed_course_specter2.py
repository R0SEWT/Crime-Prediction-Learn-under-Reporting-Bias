#!/usr/bin/env python3
"""Recomputa embeddings SPECTER2 LOCALMENTE para los papers del curso (slr-dhl).

No usamos los embeddings de Semantic Scholar (solo cubren ~20/42 del curso). En su
lugar replicamos el pipeline oficial de allenai/specter2 (cotejado con el model card
en huggingface.co/allenai/specter2 y /specter2_base):

    base    = allenai/specter2_base           (SciBERT/BERT-base, 768-d)
    adapter = allenai/specter2  (proximity)   "for general embedding purposes"
    input   = title + tokenizer.sep_token + abstract
    tokens  = padding, truncation, return_token_type_ids=False, max_length=512
    pooling = CLS  → output.last_hidden_state[:, 0, :]
    norm    = L2   (NO está en el snippet oficial; lo añadimos para kNN coseno)

El abstract se obtiene en cascada con PROCEDENCIA por paper:
    s2_bulk   ya estaba en field_corpus.json
    s2_batch  re-consulta /paper/batch fields=title,abstract (requiere S2_API_KEY)
    pdf_bronze extraído de data/bronze/<key>.txt (heurística ABSTRACT→INTRODUCTION)
    title_only no se halló abstract → embedding solo-título (gap reportado)

Requiere el venv aislado (excepción al "solo numpy/scipy" del track):
    python3 -m venv .venv-embed
    .venv-embed/bin/pip install -r scripts/analysis/requirements-embed.txt
    .venv-embed/bin/python scripts/analysis/embed_course_specter2.py

Salidas (data/silver/analysis/):
    course_embeddings.npz        ids=paperId(object), vecs=float32(N,768)  [drop-in]
    course_embeddings_meta.json  modelo/adapter/input/versiones/cobertura/procedencia
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, paths  # noqa: E402
import s2_client  # noqa: E402

MODEL_ID = "allenai/specter2_base"
ADAPTER_ID = "allenai/specter2"  # proximity (default para embeddings generales)
MAX_LENGTH = 512
BRONZE = ROOT / "data" / "bronze"

# Heurística de extracción de abstract desde texto de PDF (pdftotext -layout).
_ABS_START = re.compile(r"\b(?:a\s?b\s?s\s?t\s?r\s?a\s?c\s?t)\b[:.\s]*", re.IGNORECASE)
_ABS_END = re.compile(
    r"\b(?:I\.\s*)?introduction\b|\bindex\s+terms\b|\bkeywords?\b|\bccs\s+concepts\b"
    r"|^\s*1\.?\s+intro",
    re.IGNORECASE | re.MULTILINE,
)


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def extract_abstract_from_bronze(key: str) -> str:
    """Mejor esfuerzo: texto entre 'ABSTRACT' y 'INTRODUCTION/KEYWORDS' del PDF.

    Texto de doble columna → ruidoso; el caller marca procedencia 'pdf_bronze'.
    """
    p = BRONZE / f"{key}.txt"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = text[:9000].splitlines()
    mi = col = None
    for i, ln in enumerate(lines):
        m = _ABS_START.search(ln)
        if m:
            mi, col = i, m.start()
            break
    if mi is None:
        return ""
    # Doble columna 'ARTICLE INFO | ABSTRACT' (Elsevier/IEEE): el marcador aparece a
    # la derecha con texto a su izquierda → recortar la columna derecha por posición,
    # evitando que 'Keywords:' (columna izquierda) corte el abstract prematuramente.
    if col > 25 and lines[mi][:col].strip():
        c0 = max(0, col - 6)
        buf = []
        for ln in lines[mi + 1: mi + 40]:
            seg = ln[c0:].strip()
            if _ABS_END.search(seg):
                break
            if seg:
                buf.append(seg)
        abs = re.sub(r"\s+", " ", " ".join(buf)).strip()
    else:
        rest = "\n".join(lines[mi:])
        rest = rest[_ABS_START.search(rest).end():]
        me = _ABS_END.search(rest)
        abs = re.sub(r"\s+", " ", (rest[: me.start()] if me else rest[:2500])).strip()
    # descartar fragmentos demasiado cortos (extracción fallida)
    return abs if len(abs) >= 120 else ""


def assemble_texts(P) -> list[dict]:
    """Ensambla {paperId,key,doi,title,abstract,abstract_source} para los course_selected."""
    corpus = json.loads(P.corpus.read_text(encoding="utf-8"))
    course = [p for p in corpus["papers"] if p.get("course_selected") and p.get("paperId")]

    doi2key = s2_client.doi_to_key()
    norm = s2_client.normalize_doi

    recs = []
    for p in course:
        doi = norm(p.get("doi") or "")
        recs.append({
            "paperId": p["paperId"],
            "doi": doi,
            "key": doi2key.get(doi),
            "title": (p.get("title") or "").strip(),
            "abstract": (p.get("abstract") or "").strip(),
            "abstract_source": "s2_bulk" if (p.get("abstract") or "").strip() else None,
        })

    # 2) refetch S2 /paper/batch (title,abstract) para los que faltan
    import os
    missing = [r for r in recs if not r["abstract"]]
    if missing and os.environ.get("S2_API_KEY", "").strip():
        try:
            ids = [r["paperId"] for r in missing]
            print(f"  S2 batch refetch de abstract para {len(ids)} papers…")
            data = s2_client.s2_post(
                "/paper/batch", {"ids": ids}, params={"fields": "title,abstract"}
            )
            by_id = {d["paperId"]: d for d in (data or []) if d}
            for r in missing:
                d = by_id.get(r["paperId"]) or {}
                ab = (d.get("abstract") or "").strip()
                if ab:
                    r["abstract"], r["abstract_source"] = ab, "s2_batch"
        except Exception as e:  # noqa: BLE001
            print(f"  [aviso] refetch S2 omitido ({e}); cascada salta a pdf_bronze")
    elif missing:
        print("  [aviso] S2_API_KEY no presente; cascada salta s2_batch → pdf_bronze")

    # 3) extracción desde PDF (bronze) para los que aún faltan
    for r in recs:
        if r["abstract"] or not r["key"]:
            continue
        ab = extract_abstract_from_bronze(r["key"])
        if ab:
            r["abstract"], r["abstract_source"] = ab, "pdf_bronze"

    # 4) title_only para el resto
    for r in recs:
        if not r["abstract"]:
            r["abstract_source"] = "title_only"
    return recs


def embed(recs: list[dict], batch_size: int) -> tuple[np.ndarray, dict]:
    import torch  # import tardío: solo en el venv de embedding
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoAdapterModel.from_pretrained(MODEL_ID)
    name = model.load_adapter(ADAPTER_ID, source="hf", load_as="proximity", set_active=True)
    model.set_active_adapters(name)  # explícito: el adapter de proximity guía el forward
    model.eval()

    sep = tokenizer.sep_token
    texts = [f"{r['title']}{sep}{r['abstract']}" for r in recs]

    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt",
            return_token_type_ids=False, max_length=MAX_LENGTH,
        )
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # pooling CLS
        vecs.append(cls.cpu().numpy())
        print(f"  embebidos {min(i + batch_size, len(texts))}/{len(texts)}…")

    arr = np.vstack(vecs).astype(np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)  # L2

    pkgs = {k: pkg_version(k) for k in ("torch", "transformers", "adapters", "numpy")}
    return arr, pkgs


def write_report(meta: dict, recs: list[dict]) -> None:
    """Reporte de procedencia versionado en analysis/ (los .npz/json son gitignored)."""
    cov = meta["coverage"]
    pk = meta["packages"]
    lines = [
        "# Embeddings SPECTER2 del curso - procedencia y reproducibilidad",
        "",
        "Recómputo **local** de embeddings para los papers del curso (slr-dhl). No se usan",
        "los embeddings de Semantic Scholar (cobertura parcial). Pipeline cotejado con el",
        "model card oficial de allenai/specter2 y /specter2_base.",
        "",
        "## Configuración",
        "",
        f"- **Modelo base**: `{meta['model_id']}` ({meta['dim']}-d)",
        f"- **Adapter**: `{meta['adapter_id']}` (proximity; \"for general embedding purposes\")",
        f"- **Input**: `{meta['input_format']}`",
        f"- **max_length**: {meta['max_length']} · **pooling**: {meta['pooling'].upper()}"
        f" · **norm**: {meta['normalization'].upper()} ({meta['normalization_note']})",
        f"- **device**: {meta['device']} · **generado**: {meta['created']}",
        f"- **versiones**: torch {pk['torch']} · transformers {pk['transformers']}"
        f" · adapters {pk['adapters']} · numpy {pk['numpy']}",
        "",
        "## Cobertura del texto (título + abstract)",
        "",
        f"- **{cov['n_total']}** papers embebidos.",
        f"- **{cov['n_with_abstract']}** con abstract; **{cov['n_title_only']}** solo-título"
        " (gap reportado).",
        f"- Procedencia del abstract: {cov['by_source']}",
        "",
        "## Por paper",
        "",
        "| key | abstract | chars |",
        "|-----|----------|-------|",
    ]
    for p in sorted(meta["papers"], key=lambda x: x["abstract_source"]):
        lines.append(f"| {p['key'] or p['paperId'][:10]} | {p['abstract_source']} | {p['abstract_chars']} |")
    lines.append("")
    lines.append("> Regenerar: `.venv-embed/bin/python scripts/analysis/embed_course_specter2.py --refresh`")
    lines.append("")
    out = ROOT / "analysis" / "semantic_embeddings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ reporte → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Embeddings SPECTER2 locales del curso")
    ap.add_argument("--tag", default="", help="namespace de experimento (default: analysis)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--refresh", action="store_true", help="re-embeber aunque exista cache")
    args = ap.parse_args()

    P = paths(args.tag)
    emb_path = P.data / "course_embeddings.npz"
    meta_path = P.data / "course_embeddings_meta.json"

    if emb_path.exists() and not args.refresh:
        z = np.load(emb_path, allow_pickle=True)
        print(f"Cache: {emb_path} ({len(z['ids'])} vectores). --refresh para re-embeber.")
        return

    recs = assemble_texts(P)
    print(f"Ensamblados {len(recs)} papers del curso.")
    arr, pkgs = embed(recs, args.batch_size)

    ids = np.array([r["paperId"] for r in recs], dtype=object)
    P.data.mkdir(parents=True, exist_ok=True)
    np.savez(emb_path, ids=ids, vecs=arr)

    by_source: dict[str, int] = {}
    for r in recs:
        by_source[r["abstract_source"]] = by_source.get(r["abstract_source"], 0) + 1
    n_title_only = by_source.get("title_only", 0)

    meta = {
        "model_id": MODEL_ID,
        "adapter_id": ADAPTER_ID,
        "adapter_load": {"source": "hf", "load_as": "proximity", "set_active": True},
        "task": "proximity",
        "input_format": "title + tokenizer.sep_token + abstract",
        "input_fields": ["title", "abstract"],
        "max_length": MAX_LENGTH,
        "pooling": "cls",
        "normalization": "l2",
        "normalization_note": "L2 añadida por nosotros; no está en el snippet oficial.",
        "dim": int(arr.shape[1]),
        "device": "cpu",
        "packages": pkgs,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "doc_refs": [
            "https://huggingface.co/allenai/specter2",
            "https://huggingface.co/allenai/specter2_base",
        ],
        "coverage": {
            "n_total": len(recs),
            "n_with_abstract": len(recs) - n_title_only,
            "n_title_only": n_title_only,
            "by_source": by_source,
        },
        "papers": [
            {
                "paperId": r["paperId"], "key": r["key"], "doi": r["doi"],
                "abstract_source": r["abstract_source"],
                "abstract_chars": len(r["abstract"]),
                "title_only": r["abstract_source"] == "title_only",
            }
            for r in recs
        ],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Reporte versionado (data/silver/ es gitignored → la procedencia vive en analysis/)
    write_report(meta, recs)

    print(f"\n✓ {arr.shape[0]} embeddings ({arr.shape[1]}-d) → {emb_path}")
    print(f"✓ metadata → {meta_path}")
    print(f"  cobertura: {meta['coverage']['n_with_abstract']}/{len(recs)} con abstract; "
          f"{n_title_only} solo-título")
    print(f"  procedencia: {by_source}")


if __name__ == "__main__":
    main()
