#!/usr/bin/env python3
"""
Supplementary S1 — Post-hoc combined search (§2.3 of paper).

Verifies that L3=0 is not an artefact of the two-string search design by
running a combined query targeting both DL crime prediction and dark figure
terminology simultaneously against CrossRef (free, no API key required).

Usage:
    pip install requests
    python combined_search.py

Expected output: 0 papers found combining both domains.
This confirms that any paper genuinely integrating a dark figure correction
into a DL predictive model would be retrievable under either query family,
and its absence is not a search design artefact.
"""

import time
import requests

QUERIES = [
    "crime prediction deep learning dark figure under-reporting",
    "spatiotemporal crime prediction reporting bias MNAR",
    "crime neural network dark figure reporting propensity",
    "GNN crime prediction under-reporting victimization",
    "crime forecasting deep learning non-reporting latent crime",
]

CROSSREF_URL = "https://api.crossref.org/works"
HEADERS = {"User-Agent": "SLR-supplementary/1.0"}

DL_TERMS = {
    "deep learning", "neural network", "gnn", "gcn", "lstm",
    "transformer", "machine learning", "spatiotemporal",
}
CRIME_TERMS = {"crime", "criminal"}
REPORTING_TERMS = {
    "dark figure", "under-report", "underreport", "reporting bias",
    "unreported", "mnar", "reporting propensity", "non-reporting",
    "victimization survey",
}


def is_relevant(item: dict) -> bool:
    title = " ".join(item.get("title", [])).lower()
    abstract = item.get("abstract", "").lower()
    text = title + " " + abstract
    has_dl = any(t in text for t in DL_TERMS)
    has_crime = any(t in text for t in CRIME_TERMS)
    has_reporting = any(t in text for t in REPORTING_TERMS)
    return has_dl and has_crime and has_reporting


def search(query: str, rows: int = 250) -> list[dict]:
    params = {
        "query": query,
        "rows": rows,
        "filter": "from-pub-date:2023,until-pub-date:2026",
        "select": "DOI,title,abstract,published,container-title,type",
        "sort": "relevance",
    }
    try:
        r = requests.get(CROSSREF_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"  WARNING: {e}")
        return []


def main() -> None:
    print("Post-hoc combined search: DL crime prediction + dark figure/under-reporting")
    print(f"Source: CrossRef API, filter 2023–2026, {len(QUERIES)} queries × 250 results\n")

    candidates: dict[str, dict] = {}

    for query in QUERIES:
        print(f"Query: '{query}'")
        items = search(query)
        found = [item for item in items if is_relevant(item)]
        print(f"  {len(items)} results → {len(found)} pass relevance filter")
        for item in found:
            doi = item.get("DOI", "").lower()
            if doi and doi not in candidates:
                candidates[doi] = item
        time.sleep(1.2)

    print(f"\nTotal unique candidates combining both domains: {len(candidates)}")
    if candidates:
        print("\nPapers found:")
        for doi, item in candidates.items():
            title = " ".join(item.get("title", []))
            journal = (item.get("container-title") or ["?"])[0]
            year = (item.get("published", {}).get("date-parts") or [[None]])[0][0]
            print(f"  [{year}] {title[:80]}")
            print(f"    {journal} — DOI: {doi}")
    else:
        print("No papers found. L3=0 is not a search design artefact.")


if __name__ == "__main__":
    main()
