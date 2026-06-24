# PRISMA 2020 record flow — counts and provenance

Companion to Figure 1 of the paper. This file documents **which counts are exact and
which are reconstructed estimates**, and the source of each, in the interest of full
reporting transparency (PRISMA 2020, items 16a–16b and 24).

## Searches were conducted in May 2026 via:

| Channel | Role | Documented retrieval |
|---------|------|----------------------|
| Scopus (institutional access) | primary database | ≈ 120 records |
| ScienceDirect (institutional access) | primary database (Elsevier) | ≈ 20 records |
| CrossRef API | supplementary — IEEE/ACM venues, ISSN-targeted | ≈ 150 records |
| Semantic Scholar Academic Graph API | supplementary — venue gap filling | (folded into the above) |

Source: `knowledge/search_protocol.md` in the main repository. IEEE- and ACM-published
papers entered the corpus through the CrossRef/Semantic Scholar channel, not through a
direct IEEE Xplore or ACM Digital Library search.

## Stage counts

| PRISMA stage | Count | Status |
|--------------|-------|--------|
| Records identified | ≈ 290 | reconstructed estimate (sum of documented per-source retrieval) |
| Duplicates removed | — | reconstructed estimate |
| Records screened (title/abstract) | ≈ 280 | reconstructed estimate |
| Excluded at title/abstract | ≈ 190 | reconstructed estimate |
| Full-text assessed for eligibility | ≈ 90 | reconstructed estimate |
| Full-text excluded | — | 4 recorded with documented reasons (see below); remainder not individually logged |
| **Studies included** | **40** | **exact** (enumerated in `dois/slr_corpus_40papers.tsv`) |

### Why the intermediate counts are reconstructed, not logged

Round 1 (title/abstract) screening was performed by a single reviewer and did not retain
each excluded record individually. The intermediate stage counts are therefore
post-hoc reconstructions reported for completeness of the PRISMA flow, not logged tallies.
Two quantities are exact and verifiable: the 40 included studies and the full-text
exclusions for which a reason was recorded.

## Exact breakdown of the 40 included studies

- **By quartile:** Q1 = 33, Q2 = 7
- **By retrieval channel:** Scopus = 18, CrossRef = 16, ScienceDirect = 5, ACM Digital
  Library = 1 (also indexed in Scopus) — sums to 40
- **By year:** 2023 = 7, 2024 = 9, 2025 = 15, 2026 = 9

## Recorded full-text exclusions

Four records advancing to full text were excluded with a documented reason
(see `dois/excluded_records.tsv` for DOIs):

| cite_key | criterion | reason (short) |
|----------|-----------|----------------|
| liu2025_poi_spatial_crime | IC4 | negative-binomial regression, not ML/DL |
| briz_redon2024_bayesian_aoristic | IC4 | Bayesian logistic regression, not ML/DL |
| rezvani2024_smart_hotspot_flood | EC6 | object of study is flood risk, not crime |
| lohr2026_nibrs_crime_counts | IC4 | NIBRS sampling estimation, no ML/DL |
