#!/usr/bin/env python3
"""[TRACK ANÁLISIS] Embeddings SPECTER2 + clustering + landscape del corpus de campo.

Lee data/silver/analysis/field_corpus.json, baja embeddings SPECTER2 (768-d) vía
/paper/batch (cache), clusteriza en el espacio de embeddings (k-means coseno),
proyecta a 2D con PCA y etiqueta cada cluster con términos TF-IDF de títulos/abstracts.

Salidas:
    data/silver/analysis/embeddings.npz        (ids + vectores, cache)
    data/silver/analysis/field_clusters.json   (paperId -> cluster, x, y)
    analysis/field_landscape.md                (reporte versionado)

Uso:
    python3 scripts/analysis/cluster_field.py [--k 10] [--refresh]

Requiere S2_API_KEY. Requiere: numpy, scipy, typer, rich
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    from scipy.cluster.vq import kmeans2, whiten  # noqa: F401
    import typer
    from rich.console import Console
except ImportError:
    print("ERROR: pip install numpy scipy typer rich")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_client  # noqa: E402
from _paths import paths, ROOT  # noqa: E402

console = Console()
app = typer.Typer()

STOP = set("""the a an of and or for to in on with under via using based by from is are
be we our this that these those study studies paper approach method methods model models
results using use used new novel propose proposed framework analysis data prediction
predicting crime crimes between toward towards into over more most can which their its also
has have had not but was were will been being such than then when where while each other
""".split())

# Markup LaTeX/MathML que ensucia abstracts de algunas editoriales.
MARKUP = set("""xlink usepackage tex-math inline-formula alternatives href www mml math
mrow msub msup mi mo mn mtext documentclass begin end eqnarray label formula amsmath
amssymb wasysym upgreek setlength oddsidemargin document chi rsquo nbsp emsp ensp
""".split())
STOP |= MARKUP


def fetch_embeddings(papers: list[dict], refresh: bool, EMB: Path) -> tuple[list[str], np.ndarray]:
    """Baja embeddings SPECTER2 por paperId (batch 500). Cachea en EMB."""
    if EMB.exists() and not refresh:
        d = np.load(EMB, allow_pickle=True)
        console.print(f"[yellow]Cache embeddings: {EMB} ({len(d['ids'])} vectores). --refresh para re-bajar.[/]")
        return list(d["ids"]), d["vecs"]

    ids_all = [p["paperId"] for p in papers if p.get("paperId")]
    got_ids, vecs = [], []
    for i in range(0, len(ids_all), 500):
        chunk = ids_all[i:i + 500]
        batch = s2_client.s2_post("/paper/batch", json_body={"ids": chunk},
                                  params={"fields": "embedding.specter_v2"})
        for p in batch or []:
            if not p:
                continue
            emb = (p.get("embedding") or {}).get("vector")
            if emb:
                got_ids.append(p["paperId"])
                vecs.append(emb)
        console.print(f"[dim]  embeddings {min(i+500, len(ids_all))}/{len(ids_all)}…[/]")
    arr = np.array(vecs, dtype=np.float32)
    np.savez(EMB, ids=np.array(got_ids, dtype=object), vecs=arr)
    console.print(f"[green]✓ {len(got_ids)} embeddings → {EMB}[/]")
    return got_ids, arr


def augment_with_local_course(ids, vecs, papers, P):
    """Inyecta embeddings SPECTER2 locales (course_embeddings.npz) para los papers del
    curso que S2 no embebió, de modo que TODOS los aceptados se proyecten en el paisaje.
    Es el mismo modelo SPECTER2; la normalización L2 posterior los deja en el mismo espacio."""
    local_path = P.data / "course_embeddings.npz"
    if not local_path.exists():
        return ids, vecs
    z = np.load(local_path, allow_pickle=True)
    local = {str(i): v for i, v in zip(z["ids"], z["vecs"])}
    have = set(ids)
    missing = [p["paperId"] for p in papers
               if p.get("course_selected") and p.get("paperId")
               and p["paperId"] not in have and p["paperId"] in local]
    if not missing:
        return ids, vecs
    add = np.array([local[pid] for pid in missing], dtype=np.float32)
    console.print(f"[cyan]+ {len(missing)} embeddings locales del curso (sin specter_v2 en S2)[/]")
    return list(ids) + missing, np.vstack([vecs, add])


def top_terms(texts: list[str], n: int, all_docs: list[list[str]]) -> list[str]:
    """TF-IDF simple: términos frecuentes en el cluster, raros en el resto."""
    df = Counter()
    for toks in all_docs:
        for t in set(toks):
            df[t] += 1
    N = len(all_docs)
    tf = Counter()
    for txt in texts:
        for t in tokenize(txt):
            tf[t] += 1
    scored = {t: c * np.log(N / (1 + df[t])) for t, c in tf.items() if df[t] > 1}
    return [t for t, _ in sorted(scored.items(), key=lambda x: -x[1])[:n]]


_TAG_RE = re.compile(r"<[^>]+>")          # tags XML/HTML/MathML
_ENT_RE = re.compile(r"&[a-z]+;|&#\d+;")  # entidades
_CMD_RE = re.compile(r"\\[a-z]+")          # comandos LaTeX


def tokenize(txt: str) -> list[str]:
    txt = (txt or "").lower()
    txt = _TAG_RE.sub(" ", txt)
    txt = _ENT_RE.sub(" ", txt)
    txt = _CMD_RE.sub(" ", txt)
    toks = re.findall(r"[a-z][a-z\-]{2,}", txt)
    # descartar tokens-markup (con guion y no temáticos) y stopwords
    return [t for t in toks if t not in STOP and not (("-" in t) and any(
        m in t for m in ("formula", "graphic", "math", "link", "fig")))]


@app.command()
def main(
    k: int = typer.Option(10, "--k", help="Número de clusters (k-means)"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-bajar embeddings"),
    seed: int = typer.Option(42, "--seed"),
    tag: str = typer.Option("", "--tag", help="Namespace del experimento (ej. rq2_reporting)."),
):
    P = paths(tag)
    CORPUS, EMB, CLUSTERS, REPORT = P.corpus, P.emb, P.field_clusters, P.landscape_md
    if not CORPUS.exists():
        console.print(f"[red]Falta {CORPUS}. Corre fetch_field_corpus.py --tag {tag} primero.[/]")
        raise typer.Exit(1)
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    papers = data["papers"]
    by_id = {p["paperId"]: p for p in papers if p.get("paperId")}

    ids, vecs = fetch_embeddings(papers, refresh, EMB)
    ids, vecs = augment_with_local_course(ids, vecs, papers, P)
    if len(ids) < k:
        console.print(f"[red]Solo {len(ids)} embeddings; reduce --k.[/]")
        raise typer.Exit(1)

    # Normalizar L2 → k-means euclidiano ≈ coseno
    X = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    np.random.seed(seed)
    centroids, labels = kmeans2(X, k, minit="++", seed=seed)
    # Reasignar clusters vacíos si los hubiera (kmeans2 puede dejar vacíos)
    labels = np.asarray(labels)

    # PCA 2D (SVD sobre datos centrados)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = (Xc @ Vt[:2].T)
    # escalar a rango legible
    coords = coords / (np.abs(coords).max(axis=0) + 1e-9)

    # Documentos por cluster (para TF-IDF y reporte)
    all_docs = [tokenize(f"{by_id[i].get('title','')} {by_id[i].get('abstract','')}") for i in ids]
    out = {}
    cluster_members = {c: [] for c in range(k)}
    for idx, pid in enumerate(ids):
        c = int(labels[idx])
        out[pid] = {"cluster": c, "x": round(float(coords[idx, 0]), 4),
                    "y": round(float(coords[idx, 1]), 4)}
        cluster_members[c].append(idx)

    CLUSTERS.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Reporte
    lines = ["# Landscape del corpus de campo (track análisis)\n",
             f"_Generado por `scripts/analysis/cluster_field.py` - {len(ids)} papers con embedding, "
             f"k={k} clusters. Query: `{data['meta'].get('query','')}`_\n",
             "> Track de **análisis real**, separado del curso. No es el SLR de la tesis.\n"]
    # ordenar clusters por tamaño
    order = sorted(cluster_members, key=lambda c: -len(cluster_members[c]))
    labels_meta = {}  # cluster -> {terms, n} para la viz (leyenda + etiquetas en el mapa)
    for c in order:
        members = cluster_members[c]
        if not members:
            continue
        terms = top_terms([f"{by_id[ids[i]].get('title','')} {by_id[ids[i]].get('abstract','')}" for i in members],
                          8, all_docs)
        labels_meta[str(c)] = {"terms": terms[:4], "n": len(members)}
        n_course = sum(1 for i in members if by_id[ids[i]].get("course_selected"))
        # más influyentes del cluster
        infl = sorted(members, key=lambda i: -(by_id[ids[i]].get("citationCount") or 0))[:3]
        top_titles = [f"{by_id[ids[i]].get('title','')[:80]} ({by_id[ids[i]].get('citationCount') or 0} cit)" for i in infl]
        lines.append(f"\n## Cluster {c} - {len(members)} papers "
                     f"({n_course} del curso)\n")
        lines.append(f"**Términos:** {', '.join(terms)}\n")
        lines.append("**Más citados:**\n" + "\n".join(f"- {t}" for t in top_titles) + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    # etiquetas de cluster para la viz (términos top reproducibles)
    CLUSTERS.with_name("cluster_labels.json").write_text(
        json.dumps(labels_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    n_course = sum(1 for pid in ids if by_id[pid].get("course_selected"))
    console.print(f"[green]✓ {len(ids)} papers clusterizados (k={k}) → {CLUSTERS}[/]")
    console.print(f"  course_selected con embedding: {n_course}")
    console.print(f"  Reporte: {REPORT.relative_to(ROOT)}")
    # distribución de course papers por cluster
    cc = Counter(out[pid]["cluster"] for pid in ids if by_id[pid].get("course_selected"))
    console.print(f"  Distribución curso por cluster: {dict(sorted(cc.items()))}")


if __name__ == "__main__":
    app()
