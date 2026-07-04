#!/usr/bin/env python3
"""Cliente compartido de la Semantic Scholar Academic Graph API.

Centraliza: lectura de la API key, rate-limiting (S2 permite 1 req/s con key),
retry en 429/5xx, y helpers que leen el corpus de papers (knowledge/papers/*.md)
para deduplicar descubrimientos y mapear DOI <-> key.

La API key se lee de la variable de entorno S2_API_KEY (NO se guarda en el repo):

    export S2_API_KEY="s2k-..."   # en ~/.zshrc

Reutilizado por:
    - scripts/journal_search.py        (descubrimiento)
    - scripts/build_citation_edges.py  (grafo de citas)

Requiere: pip install requests pyyaml
"""

from __future__ import annotations

import os
import sys
import time
import re
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
MIN_INTERVAL = 1.1  # s entre requests - el rate limit con key es 1 req/s

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "knowledge" / "papers"

_last_call = 0.0


def get_key() -> str:
    """Devuelve la API key desde el entorno, o aborta con instrucciones."""
    key = os.environ.get("S2_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: falta la variable de entorno S2_API_KEY.\n"
            "  Agrega a tu ~/.zshrc:\n"
            '    export S2_API_KEY="s2k-..."\n'
            "  Luego: source ~/.zshrc  (o abre una terminal nueva).",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _throttle() -> None:
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_call = time.monotonic()


def _request(method: str, path: str, *, params=None, json_body=None,
             max_retries: int = 6):
    """Request rate-limited con retry exponencial en 429/5xx.

    El endpoint de S2 throttlea agresivamente (429) incluso con key, porque el
    rate limit es compartido. Por eso el backoff es paciente (3,6,12,24,48 s).
    """
    headers = {"x-api-key": get_key(), "User-Agent": "infelix-slr/1.0 (research)"}
    url = f"{BASE_URL}{path}"
    backoff = 3.0
    for attempt in range(max_retries):
        _throttle()
        try:
            r = requests.request(method, url, headers=headers, params=params,
                                 json=json_body, timeout=30)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries - 1:
                r.raise_for_status()
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"S2 request agotó reintentos: {method} {url}")


def s2_get(path: str, params: dict | None = None):
    """GET contra la S2 graph API. path empieza con '/'."""
    return _request("GET", path, params=params)


def s2_post(path: str, json_body, params: dict | None = None):
    """POST contra la S2 graph API (ej. /paper/batch)."""
    return _request("POST", path, params=params, json_body=json_body)


def s2_bulk_search(query: str, fields: str, *, cap: int | None = None,
                   **extra) -> tuple[int, list[dict]]:
    """Itera /paper/search/bulk vía `token` de continuación.

    Devuelve (total_reportado, papers). Para si alcanza `cap`.
    """
    out: list[dict] = []
    params = {"query": query, "fields": fields, **extra}
    total = None
    token = None
    while True:
        if token:
            params["token"] = token
        data = s2_get("/paper/search/bulk", params)
        if total is None:
            total = data.get("total", 0)
        out.extend(data.get("data") or [])
        token = data.get("token")
        if not token or (cap is not None and len(out) >= cap):
            break
    if cap is not None:
        out = out[:cap]
    return total or 0, out


def s2_paginate(path: str, fields: str, *, page_size: int = 1000,
                max_items: int | None = None) -> list[dict]:
    """Itera endpoints con paginación offset/next (references, citations).

    Devuelve la lista acumulada de items (`data`).
    """
    out: list[dict] = []
    offset = 0
    while True:
        data = s2_get(path, {"fields": fields, "offset": offset, "limit": page_size})
        batch = data.get("data") or []
        out.extend(batch)
        nxt = data.get("next")
        if nxt is None or not batch:
            break
        offset = nxt
        if max_items is not None and len(out) >= max_items:
            return out[:max_items]
    return out


# ── Helpers de corpus ─────────────────────────────────────────────────────

_DOI_RE = re.compile(r'^doi:\s*"?([^"\n]+?)"?\s*$', re.MULTILINE)
_KEY_RE = re.compile(r'^key:\s*"?([^"\n]+?)"?\s*$', re.MULTILINE)
_SCREEN_RE = re.compile(r'^screening:\s*"?([^"\n]+?)"?\s*$', re.MULTILINE)


def normalize_doi(doi: str) -> str:
    """Normaliza un DOI para comparación: minúsculas, sin prefijo URL/doi:."""
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def _iter_frontmatter():
    """Yield (key, doi_normalizado, screening) por cada stub con frontmatter."""
    for path in sorted(PAPERS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        try:
            fm = text[3:text.index("---", 3)]
        except ValueError:
            continue
        km = _KEY_RE.search(fm)
        dm = _DOI_RE.search(fm)
        sm = _SCREEN_RE.search(fm)
        key = km.group(1).strip() if km else path.stem
        doi = normalize_doi(dm.group(1)) if dm else ""
        screening = sm.group(1).strip() if sm else ""
        yield key, doi, screening


def corpus_dois(only_accepted: bool = False) -> set[str]:
    """DOIs normalizados ya presentes en el corpus (para deduplicar)."""
    out = set()
    for _key, doi, screening in _iter_frontmatter():
        if not doi:
            continue
        if only_accepted and screening != "accepted":
            continue
        out.add(doi)
    return out


def doi_to_key(only_accepted: bool = False) -> dict[str, str]:
    """Mapa DOI normalizado -> key del corpus."""
    out = {}
    for key, doi, screening in _iter_frontmatter():
        if not doi:
            continue
        if only_accepted and screening != "accepted":
            continue
        out[doi] = key
    return out


def key_to_doi(only_accepted: bool = False) -> dict[str, str]:
    """Mapa key -> DOI normalizado del corpus."""
    out = {}
    for key, doi, screening in _iter_frontmatter():
        if not doi:
            continue
        if only_accepted and screening != "accepted":
            continue
        out[key] = doi
    return out


if __name__ == "__main__":
    # Smoke test rápido: cuenta corpus y hace 1 búsqueda mínima.
    print(f"Corpus: {len(corpus_dois())} DOIs ({len(corpus_dois(only_accepted=True))} accepted)")
    data = s2_get("/paper/search", {"query": "crime prediction", "limit": 1,
                                     "fields": "title,year,venue"})
    print("S2 OK:", data.get("data"))
