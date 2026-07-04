#!/usr/bin/env python3
"""[TRACK ANÁLISIS] Construye el corpus de campo vía Semantic Scholar bulk search.

Track de ANÁLISIS REAL - separado del track del curso (knowledge/papers/, doc/).
Baja cientos de papers del área (sin filtro de cuartil; es paisaje, no el SLR del
curso), marca los que están en el corpus curado (course_selected) y guarda metadata
+ ids para clustering por embeddings.

Salida: data/silver/analysis/field_corpus.json  (regenerable; cache + --refresh)

Uso:
    python3 scripts/analysis/fetch_field_corpus.py --preset broad-crime
    python3 scripts/analysis/fetch_field_corpus.py --query "..." --cap 500
    python3 scripts/analysis/fetch_field_corpus.py --preset rq2-reporting --year-from 2015

Requiere S2_API_KEY en el entorno. Requiere: pip install requests typer rich pyyaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import typer
    from rich.console import Console
except ImportError:
    print("ERROR: pip install typer rich requests pyyaml")
    sys.exit(1)

# s2_client vive en scripts/ (un nivel arriba); _paths en el mismo dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_client  # noqa: E402
from _paths import paths  # noqa: E402

console = Console()
app = typer.Typer(help="Corpus de campo para el track de análisis (S2 bulk).")

# Presets de query (sintaxis bulk de S2: | = OR, + = AND, "..." = frase)
PRESETS = {
    "dl-crime": '(crime prediction | crime forecasting | crime hotspot) + '
                '(deep learning | "neural network" | "graph neural network" | '
                'LSTM | transformer | CNN)',
    "broad-crime": '(crime prediction | crime forecasting | crime hotspot | '
                   '"crime mapping" | "criminal risk" | "spatiotemporal crime" | '
                   '"urban crime") + (model | learning | prediction | forecasting | neural)',
    # Anclado en crimen para excluir la enorme literatura médica de "reporting bias"
    # (publication bias en ensayos clínicos, self-reported health, etc.).
    "rq2-reporting": '("dark figure of crime" | "crime under-reporting" | '
                     '"under-reporting of crime" | "unreported crime" | '
                     '"victimization survey" | "police-recorded crime" | '
                     '"crime reporting" | "reporting of crime" | '
                     '"willingness to report" | "crime victimization") + '
                     '(crime | criminal | victimization | police | offending | delinquency)',
}

# Nota: 'tldr' NO está soportado por bulk search (da 400) - se trae luego vía
# /paper/batch en cluster_field.py si se necesita.
FIELDS = ("title,abstract,year,venue,publicationVenue,externalIds,citationCount,"
          "influentialCitationCount,referenceCount,fieldsOfStudy,publicationDate")


@app.command()
def main(
    preset: str = typer.Option(None, "--preset", "-p",
                               help=f"Preset de query: {', '.join(PRESETS)}"),
    query: str = typer.Option(None, "--query", "-q", help="Query libre (sintaxis bulk S2)"),
    year_from: int = typer.Option(2010, "--year-from", "-y", help="Desde este año (>=)"),
    cap: int = typer.Option(800, "--cap", help="Máx. papers a bajar"),
    sort: str = typer.Option("", "--sort", help="Orden S2 bulk, ej. 'citationCount:desc' "
                             "o 'publicationDate:desc'. Vacío = orden por defecto."),
    force: bool = typer.Option(False, "--force", help="Bajar aunque total > cap"),
    refresh: bool = typer.Option(False, "--refresh", help="Ignorar cache y re-consultar"),
    remark: bool = typer.Option(False, "--remark", help="Re-marcar course_selected sobre el "
                                "field_corpus.json existente (solo aceptados) SIN re-consultar S2."),
    tag: str = typer.Option("", "--tag", help="Namespace del experimento (ej. rq2_reporting). "
                            "Vacío = broad-crime por defecto."),
):
    """Baja el corpus de campo y marca los papers del corpus curado (course_selected)."""
    if not preset and not query:
        console.print("[red]Indica --preset o --query.[/]")
        console.print(f"[dim]Presets: {', '.join(PRESETS)}[/]")
        raise typer.Exit(1)
    q = query or PRESETS.get(preset)
    if q is None:
        console.print(f"[red]Preset '{preset}' desconocido. Opciones: {', '.join(PRESETS)}[/]")
        raise typer.Exit(1)

    P = paths(tag)
    OUT = P.corpus
    P.data.mkdir(parents=True, exist_ok=True)

    # Modo re-marcado: recomputa course_selected (solo aceptados) sobre el corpus ya
    # descargado, sin tocar S2 ni los clusters. Útil cuando cambia el screening.
    if remark:
        if not OUT.exists():
            console.print(f"[red]No existe {OUT}. Corre la descarga primero (sin --remark).[/]")
            raise typer.Exit(1)
        data = json.loads(OUT.read_text(encoding="utf-8"))
        accepted = s2_client.corpus_dois(only_accepted=True)
        n = 0
        for row in data["papers"]:
            is_course = bool(row.get("doi") and row["doi"] in accepted)
            row["course_selected"] = is_course
            n += is_course
        data.setdefault("meta", {})
        data["meta"]["course_selected"] = n
        data["meta"]["course_total"] = len(accepted)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]✓ Re-marcado: {n} course_selected (de {len(accepted)} aceptados) → {OUT}[/]")
        raise typer.Exit(0)

    if OUT.exists() and not refresh:
        console.print(f"[yellow]Cache existe: {OUT}. Usa --refresh para re-bajar.[/]")
        raise typer.Exit(0)

    # Preview de total
    total, _ = s2_client.s2_bulk_search(q, "paperId", year=f"{year_from}-", cap=1)
    console.print(f"[bold]Query:[/] {q}")
    console.print(f"[bold]Total reportado por S2 ({year_from}+):[/] {total}")
    if total > cap and not force:
        console.print(f"[yellow]Total ({total}) > cap ({cap}). Subí --cap o usa --force "
                      f"para bajar solo los primeros {cap}.[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Bajando hasta {cap} papers" + (f" (orden {sort})" if sort else "") + "…[/]")
    extra = {"year": f"{year_from}-"}
    if sort:
        extra["sort"] = sort
    _, papers = s2_client.s2_bulk_search(q, FIELDS, cap=cap, **extra)

    course = s2_client.corpus_dois(only_accepted=True)

    def to_row(p, is_course):
        doi = (p.get("externalIds") or {}).get("DOI", "")
        doi_n = s2_client.normalize_doi(doi) if doi else ""
        return {
            "paperId": p.get("paperId"),
            "doi": doi_n,
            "title": p.get("title"),
            "abstract": p.get("abstract"),
            "tldr": (p.get("tldr") or {}).get("text"),
            "year": p.get("year"),
            "venue": p.get("venue"),
            "citationCount": p.get("citationCount"),
            "influentialCitationCount": p.get("influentialCitationCount"),
            "referenceCount": p.get("referenceCount"),
            "fieldsOfStudy": p.get("fieldsOfStudy"),
            "course_selected": is_course,
        }

    rows = []
    seen_dois = set()
    for p in papers:
        doi_n = s2_client.normalize_doi((p.get("externalIds") or {}).get("DOI", "") or "")
        rows.append(to_row(p, doi_n in course))
        if doi_n:
            seen_dois.add(doi_n)

    # Unión con el corpus curado: garantizar que los 44 estén presentes y marcados,
    # aunque la query del campo no los traiga (RQ2/RQ3 caen fuera de "prediction").
    missing = sorted(course - seen_dois)
    if missing:
        console.print(f"[dim]Trayendo {len(missing)} paper(s) curados ausentes vía batch…[/]")
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            batch = s2_client.s2_post(
                "/paper/batch",
                json_body={"ids": [f"DOI:{d}" for d in chunk]},
                params={"fields": FIELDS},
            )
            for p in batch or []:
                if p:
                    rows.append(to_row(p, True))

    marked = sum(1 for r in rows if r["course_selected"])
    meta = {"query": q, "year_from": year_from, "total_reported": total,
            "downloaded": len(rows), "course_selected": marked,
            "course_total": len(course), "preset": preset, "sort": sort, "cap": cap}
    OUT.write_text(json.dumps({"meta": meta, "papers": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✓ {len(rows)} papers → {OUT}[/]")
    console.print(f"  course_selected marcados: {marked} (de {len(course)} en el corpus curado)")
    console.print("Siguiente: python3 scripts/analysis/cluster_field.py")


if __name__ == "__main__":
    app()
