#!/usr/bin/env python3
"""[TRACK ANÁLISIS] Núcleo extendido por snowballing + coupling + co-citación.

Desde los papers del curso (course_selected en field_corpus.json):
  1. Baja refs completas (/references) y citers (/citations) de cada uno - esto
     ARREGLA el sub-conteo del batch (hoy 27/38 papers con 0 refs).
  2. Snowballing: refs compartidas (backward, seminales) + citers frecuentes
     (forward, relevantes) → candidatos rankeados → extiende el núcleo a --size.
  3. Computa, en el núcleo: red de citas directa, bibliographic coupling
     (refs compartidas, Salton) y co-citación (citers compartidos), y clusteriza.

Salidas (regenerables, cache incremental):
  data/silver/analysis/refs_full.json, citers.json
  data/silver/analysis/{coupling,cocitation,citation_net,core_clusters}.json
  analysis/bibliometrics.md  (reporte versionado)

Uso: python3 scripts/analysis/build_core_ext.py [--size 90] [--clusters 6] [--refresh]
Requiere S2_API_KEY. Requiere: numpy scipy typer rich
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    import typer
    from rich.console import Console
except ImportError:
    print("ERROR: pip install numpy scipy typer rich")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_client  # noqa: E402
from _paths import paths, ROOT  # noqa: E402

CITER_CAP = 2000  # máx. citers por paper (acota costo/throttling en papers muy citados)

console = Console()
app = typer.Typer()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fetch_links(pids: list[str], cache_path: Path, kind: str, refresh: bool) -> dict:
    """Baja refs o citers (paperIds) por paper, con cache incremental por paperId."""
    cache = {} if refresh else _load(cache_path)
    endpoint = "references" if kind == "refs" else "citations"
    wrap = "citedPaper" if kind == "refs" else "citingPaper"
    todo = [p for p in pids if p not in cache]
    failed = 0
    for n, pid in enumerate(todo, 1):
        try:
            items = s2_client.s2_paginate(
                f"/paper/{pid}/{endpoint}", "paperId",
                page_size=1000, max_items=None if kind == "refs" else CITER_CAP)
            cache[pid] = [it[wrap]["paperId"] for it in items
                          if it.get(wrap) and it[wrap].get("paperId")]
        except Exception as e:
            # Un paper muy citado puede agotar reintentos por throttling: lo
            # saltamos (lista vacía) en vez de abortar todo el análisis.
            cache[pid] = []
            failed += 1
            console.print(f"[yellow]  {kind}: saltado {pid[:12]} ({type(e).__name__})[/]")
        if n % 10 == 0 or n == len(todo):
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            console.print(f"[dim]  {kind}: {n}/{len(todo)} nuevos ({failed} saltados)…[/]")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def salton(shared: int, a: int, b: int) -> float:
    return shared / (np.sqrt(a * b) + 1e-9) if a and b else 0.0


@app.command()
def main(
    size: int = typer.Option(90, "--size", help="Tamaño del núcleo extendido"),
    sim_threshold: float = typer.Option(0.1, "--sim-threshold",
                                        help="Similitud mínima para agrupar (clustering por umbral)"),
    alpha: float = typer.Option(0.5, "--alpha", help="Peso de coupling vs co-citación"),
    seed_mode: str = typer.Option("course", "--seed-mode",
                                  help="Semillas del snowball: 'course' (papers del curso) "
                                  "o 'topcited' (más citados del campo)."),
    refresh: bool = typer.Option(False, "--refresh"),
    tag: str = typer.Option("", "--tag", help="Namespace del experimento (ej. rq2_reporting)."),
):
    P = paths(tag)
    CORPUS, REFS, CITERS, REPORT, ADIR = P.corpus, P.refs, P.citers, P.biblio_md, P.data
    if not CORPUS.exists():
        console.print(f"[red]Falta {CORPUS}. Corre fetch_field_corpus.py --tag {tag}.[/]")
        raise typer.Exit(1)
    field = json.loads(CORPUS.read_text(encoding="utf-8"))["papers"]
    by_id = {p["paperId"]: p for p in field if p.get("paperId")}

    # Semillas del snowballing según --seed-mode:
    #  course   → papers del curso (course_selected): núcleo de la tesis.
    #  topcited → papers más citados del campo: núcleo seminal del landscape
    #             (apropiado para experimentos como rq2_reporting).
    if seed_mode == "topcited":
        seeds = [p["paperId"] for p in sorted(
                 field, key=lambda p: -(p.get("citationCount") or 0))
                 if p.get("paperId")][:size]
        console.print(f"[bold]Semillas (top-cited del campo):[/] {len(seeds)} papers")
    else:
        seeds = [p["paperId"] for p in field if p.get("course_selected") and p.get("paperId")]
        console.print(f"[bold]Semillas (curso):[/] {len(seeds)} papers con paperId")

    # 1) refs + citers de las semillas
    refs = fetch_links(seeds, REFS, "refs", refresh)
    citers = fetch_links(seeds, CITERS, "citers", refresh)
    nz = sum(1 for p in seeds if refs.get(p))
    console.print(f"  refs no vacías: {nz}/{len(seeds)}")

    # 2) Snowballing: frecuencia de refs (backward) y citers (forward)
    back = Counter(r for p in seeds for r in refs.get(p, []))
    fwd = Counter(c for p in seeds for c in citers.get(p, []))
    score = Counter()
    seedset0 = set(seeds)
    for pid, n in back.items():
        if pid not in seedset0:
            score[pid] += n          # seminal: citado por muchas semillas
    for pid, n in fwd.items():
        if pid not in seedset0:
            score[pid] += n          # relevante: cita a muchas semillas
    n_add = max(0, size - len(seeds))
    candidates = [pid for pid, _ in score.most_common() if score[pid] >= 2][:n_add]

    # Metadata de candidatos (batch; algunos fuera del field corpus)
    need_meta = [c for c in candidates if c not in by_id]
    for i in range(0, len(need_meta), 100):
        chunk = need_meta[i:i + 100]
        batch = s2_client.s2_post("/paper/batch", json_body={"ids": chunk},
                                  params={"fields": "title,year,venue,externalIds,"
                                          "citationCount,influentialCitationCount"})
        for p in batch or []:
            if p and p.get("paperId"):
                p["course_selected"] = False
                by_id[p["paperId"]] = p

    core = seeds + [c for c in candidates if c in by_id]
    console.print(f"[bold]Núcleo extendido:[/] {len(core)} ({len(seeds)} semilla + "
                  f"{len(core)-len(seeds)} snowball)")

    # refs + citers de los candidatos añadidos
    added = [c for c in core if c not in set(seeds)]
    if added:
        refs = fetch_links(added, REFS, "refs", False)   # cache incremental, no refresh
        citers = fetch_links(added, CITERS, "citers", False)

    # 3) Matrices coupling / co-citación / red de citas
    refset = {p: set(refs.get(p, [])) for p in core}
    citeset = {p: set(citers.get(p, [])) for p in core}
    coreset = set(core)
    idx = {p: i for i, p in enumerate(core)}
    N = len(core)
    coupling = np.zeros((N, N)); cocite = np.zeros((N, N))
    cite_edges = []
    for a in range(N):
        pa = core[a]
        # red de citas directa: pa cita pb si pb ∈ refs(pa)
        for pb in refset[pa] & coreset:
            if pb != pa:
                cite_edges.append({"source": pa, "target": pb})
        for b in range(a + 1, N):
            pb = core[b]
            sc = len(refset[pa] & refset[pb])
            so = len(citeset[pa] & citeset[pb])
            coupling[a, b] = coupling[b, a] = salton(sc, len(refset[pa]), len(refset[pb]))
            cocite[a, b] = cocite[b, a] = salton(so, len(citeset[pa]), len(citeset[pb]))

    # 4) Clustering sobre similitud combinada (solo nodos con enlaces de cita).
    #    Muchos papers recientes no tienen refs/citers en S2 → similitud 0 con todo;
    #    forzar maxclust colapsa todo en un cluster. En su lugar: agrupar por umbral
    #    de distancia solo los nodos conectados; los aislados → cluster 0.
    S = alpha * coupling + (1 - alpha) * cocite  # solo off-diagonal
    labels = np.zeros(N, dtype=int)  # 0 = sin enlaces de cita (aislado)
    connected = np.where(S.sum(axis=1) > 1e-9)[0]
    if len(connected) >= 3:
        Sub = S[np.ix_(connected, connected)]
        Dsub = 1.0 - Sub
        np.fill_diagonal(Dsub, 0.0)
        Dsub = (Dsub + Dsub.T) / 2
        sub = fcluster(linkage(squareform(Dsub, checks=False), method="average"),
                       t=1.0 - sim_threshold, criterion="distance")
        for k, i in enumerate(connected):
            labels[i] = int(sub[k])
    console.print(f"  conectados por citas: {len(connected)}/{N} | "
                  f"clusters: {len(set(labels[labels > 0]))} (+ aislados)")

    # Salidas JSON
    ADIR.mkdir(parents=True, exist_ok=True)
    (ADIR / "citation_net.json").write_text(json.dumps(cite_edges, indent=2), encoding="utf-8")

    def top_pairs(M, n=15):
        pairs = [(core[a], core[b], float(M[a, b]))
                 for a in range(N) for b in range(a + 1, N) if M[a, b] > 0]
        return sorted(pairs, key=lambda x: -x[2])[:n]

    coup_pairs = top_pairs(coupling); coc_pairs = top_pairs(cocite)
    (ADIR / "coupling.json").write_text(json.dumps(
        [{"a": a, "b": b, "w": round(w, 3)} for a, b, w in top_pairs(coupling, 200)],
        indent=2), encoding="utf-8")
    (ADIR / "cocitation.json").write_text(json.dumps(
        [{"a": a, "b": b, "w": round(w, 3)} for a, b, w in top_pairs(cocite, 200)],
        indent=2), encoding="utf-8")
    (ADIR / "core_clusters.json").write_text(json.dumps(
        {core[i]: int(labels[i]) for i in range(N)}, indent=2), encoding="utf-8")

    # 5) Reporte
    def title(pid):
        return (by_id.get(pid, {}).get("title") or pid)[:75]
    def cc(pid):
        return by_id.get(pid, {}).get("citationCount") or 0
    def icc(pid):
        return by_id.get(pid, {}).get("influentialCitationCount") or 0

    n_clusters = len(set(int(x) for x in labels if x > 0))
    n_course_core = sum(1 for p in core if by_id.get(p, {}).get("course_selected"))
    L = ["# Bibliometría del núcleo extendido (track análisis)\n",
         f"_`scripts/analysis/build_core_ext.py` (seed-mode={seed_mode}) - núcleo {N} "
         f"({len(seeds)} semilla + {N-len(seeds)} snowball; {n_course_core} del curso), "
         f"{len(cite_edges)} citas directas intra-núcleo, "
         f"{n_clusters} clusters de cita + aislados (α={alpha})._\n",
         "> Track de **análisis real**, separado del SLR del curso.\n",
         f"\n## Red de citas directa\n{len(cite_edges)} edges intra-núcleo "
         f"(dirigidos, quién-cita-a-quién; {nz}/{len(seeds)} semillas con refs en S2).\n",
         "\n## Top 15 por citas influyentes (S2 influentialCitationCount)\n"]
    for pid in sorted(core, key=lambda p: -icc(p))[:15]:
        mark = "★" if by_id.get(pid, {}).get("course_selected") else " "
        L.append(f"- {mark} **{icc(pid)}** infl / {cc(pid)} cit - {title(pid)}")
    L.append("\n## Top 15 pares por bibliographic coupling (refs compartidas)\n")
    for a, b, w in coup_pairs:
        L.append(f"- {w:.2f} - {title(a)} ↔ {title(b)}")
    L.append("\n## Top 15 pares por co-citación (citados juntos)\n")
    for a, b, w in coc_pairs:
        L.append(f"- {w:.2f} - {title(a)} ↔ {title(b)}")
    # La similitud de citas es block-sparse → micro-comunidades (pares/tríos)
    # muy cohesionadas. Listamos las de ≥3 miembros; el resto se resume.
    # Para la vista macro-temática usar los clusters de embeddings (field_landscape.md).
    sizes = Counter(int(x) for x in labels)
    big = [c for c in sizes if c != 0 and sizes[c] >= 3]
    pairs = [c for c in sizes if c != 0 and sizes[c] == 2]
    L.append("\n## Micro-comunidades de cita (coupling+co-citación)\n")
    L.append(f"La similitud de citas es block-sparse: {len(big)} comunidades de ≥3, "
             f"{len(pairs)} pares, {sizes.get(0,0)} aislados. (Para temas macro ver "
             f"`field_landscape.md`, clustering por embeddings.)\n")
    for c in sorted(big, key=lambda c: -sizes[c]):
        members = [core[i] for i in range(N) if labels[i] == c]
        ncourse = sum(1 for p in members if by_id.get(p, {}).get("course_selected"))
        reps = sorted(members, key=lambda p: -cc(p))[:5]
        L.append(f"\n### Comunidad {c} - {len(members)} papers ({ncourse} del curso)")
        for p in reps:
            L.append(f"- {title(p)} ({cc(p)} cit)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    console.print(f"[green]✓ Núcleo analizado.[/] Citas directas: {len(cite_edges)} | "
                  f"reporte: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
