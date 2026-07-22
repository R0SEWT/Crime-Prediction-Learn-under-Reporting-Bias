# Supplementary Materials

**Papers:**

1. **S1 — SLR:** *Deep Learning and Spatiotemporal Modelling for Urban Crime Prediction: A Systematic Review of Architectures, Heterogeneous Data Fusion, and the Dark Figure Gap (2023–mid 2026)*
2. **Companion (Bayesian de-biasing, preprint):** *Estimating Latent Urban Crime Risk Under-Reporting Bias: A Bayesian Small Area Approach for Lima Metropolitana* — see [`companion/`](companion/) for the replication package (scripts, crosswalk, derived artefacts, canonical-numbers registry).
3. **SIMBig 2026 (empirical):** *What Can Fine-Grained Crime Prediction Learn under Reporting Bias? Label Circularity, Cross-Modal Fusion, and Cross-City Transfer in Peru* — see [`simbig/`](simbig/) for the full experimental package (manuscript → script/report/metrics map).

**Authors:** Rody Sebastian Vilchez Marin · Christian Aaron Velasquez Borasino  
**Affiliation:** Universidad Peruana de Ciencias Aplicadas (UPC), Lima, Peru

---

## Contents

### `prisma/` — PRISMA S1 verifiable record (SLR)

| File | Description |
|------|-------------|
| `prisma_s1.md` | Tally verification (manuscript vs extraction stubs), included/excluded tables with provenance; regenerable via `build_prisma_s1.py` in the main pipeline |
| `prisma_s1_included.csv` | The 40 included studies with DOI, journal, quartile, year, RQs |
| `prisma_s1_excluded.csv` | The 4 full-text exclusions with documented IC/EC reason |

### `screening/` — Dual-screening reliability (SLR §2)

| File | Description |
|------|-------------|
| `screening_kappa.md` | Inter-rater agreement analysis (Cohen's κ) for the dual re-screening |
| `screening_kappa.json` | Raw κ computation output |
| `screening_universe.csv` | Recoverable screening universe |
| `christian_screening_template.csv` / `screening_adjudication.csv` | Second-reviewer blind template and adjudicated disagreements |

### `dois/` — Paper identifier lists

| File | Description | Papers |
|------|-------------|--------|
| `slr_corpus_40papers.tsv` | The 40 SLR corpus papers (Q1/Q2 journals, 2023–mid 2026) with cite keys, DOIs, journals, quartiles, and research questions addressed | 40 |
| `field_landscape_corpus.tsv` | Broad crime prediction landscape used for bibliometric clustering (§4.3) — papers with SPECTER2 embeddings from Semantic Scholar | 1,107 |
| `reporting_landscape_corpus.tsv` | Reporting bias / dark figure landscape used for overlap analysis (§4.4) — papers with SPECTER2 embeddings | 1,184 |
| `extended_core_90papers.tsv` | Extended citation core: 40 SLR papers + 50 papers from citation snowballing (§4.2) | 90 |
| `excluded_records.tsv` | Full-text exclusions with documented reasons (DOI, stage, IC/EC criterion) | 4 |

All TSV files are UTF-8, tab-separated, with a header row.

### PRISMA flow

`prisma_counts.md` documents the PRISMA 2020 record flow (Figure 1 of the paper),
stating which stage counts are exact and which are reconstructed estimates, with the
source of each. See it for the search-channel breakdown and the exact composition of the
40 included studies.

### `code/` — Reproduction scripts

| File | Description |
|------|-------------|
| `cluster_robustness.py` | k-sensitivity analysis for §4.4: clusters `reporting_landscape_corpus` at k∈{7,10,12,15} and counts SLR papers in reporting-bias clusters. Expected result: 3 papers (0.25%) for k∈{10,12,15}. |
| `combined_search.py` | Post-hoc combined search (§2.3): CrossRef queries combining DL crime prediction + dark figure terms. Expected result: 0 additional papers, confirming L3=0 is not a search design artefact. |
| `fetch_field_corpus.py` | Builds the broad field corpus from Semantic Scholar bulk search (presets, cap) |
| `cluster_field.py` | SPECTER2 k-means clustering + PCA projection of the field corpus |
| `build_core_ext.py` | Citation snowballing → extended core; bibliographic coupling + co-citation |
| `embed_course_specter2.py` | Local SPECTER2 embeddings for the course corpus (torch env: `requirements-embed.txt`) |
| `build_semantic_edges.py` | k-NN semantic edges over the course embeddings (numpy only) |
| `s2_client.py` | Shared Semantic Scholar client (1 req/s, retry on 429; key via `S2_API_KEY` env) |

---

## Reproducing the clustering analysis

### Requirements

```bash
pip install numpy scipy requests
```

### Step 1: Obtain SPECTER2 embeddings

Embeddings are not distributed here (1,184 × 768-dim float32 ≈ 3.4 MB binary).  
To reproduce, you need a [Semantic Scholar API key](https://www.semanticscholar.org/product/api):

```bash
export S2_API_KEY="your-key-here"
```

Then fetch embeddings for each paper ID in `reporting_landscape_corpus.tsv`:

```python
import requests, numpy as np

ids = [row["s2_paper_id"] for row in tsv_reader("dois/reporting_landscape_corpus.tsv")]
# POST to https://api.semanticscholar.org/graph/v1/paper/batch
# fields=embedding.specter_v2
# Rate limit: 1 req/s with key, batches of 500
```

Save the result as `dois/reporting_embeddings.npz` with keys `ids` and `vecs`.

### Step 2: Run robustness check

```bash
python code/cluster_robustness.py
```

Expected output:
```
   k    Reporting clusters    SLR in reporting       %
   7               [...]                   20    1.69%
  10               [...]                    3    0.25%
  12               [...]                    3    0.25%
  15               [...]                    3    0.25%
```

### Step 3: Verify combined search result

```bash
python code/combined_search.py
```

Expected output: `Total unique candidates combining both domains: 0`

---

## Search queries (from §2.3)

**Main query (RQ1):**
```
("crime prediction" OR "criminal risk" OR "crime risk")
AND ("deep learning" OR "machine learning" OR "neural network")
AND ("spatiotemporal" OR "spatio-temporal" OR "spatial")
```

**RQ2 query (reporting bias):**
```
("under-reporting" OR "underreporting" OR "dark figure" OR "crime reporting bias"
OR "victimization survey" OR "victimisation survey" OR "reporting propensity"
OR "non-reporting" OR "police data validity" OR "measurement error"
OR "latent crime" OR "crime data reliability")
AND ("crime" OR "criminal")
```

**RQ3 query (heterogeneous data):**
```
("multimodal" OR "multi-source" OR "heterogeneous data")
AND ("crime" OR "urban safety")
AND ("deep learning" OR "GNN" OR "GCN" OR "LSTM")
```

**Post-hoc combined query (§2.3 robustness check):**
```
("crime prediction" OR "criminal risk")
AND ("deep learning" OR "neural network" OR "GNN")
AND ("dark figure" OR "under-reporting" OR "reporting bias" OR "MNAR" OR "victimization survey")
```

Databases searched: Scopus, IEEE Xplore, ACM Digital Library, ScienceDirect (May 2026).  
Post-hoc combined query: CrossRef API (May 2026).

---

## Bibliometric methods

- **Embedding model:** SPECTER2 (`allenai/specter2_base` + proximity adapter `allenai/specter2`), input `title[SEP]abstract`, max_length=512, CLS pooling, L2-normalised. Papers missing S2 embeddings (22/40 SLR papers) were recomputed locally using the same pipeline.
- **Clustering:** k-means on 768-dim L2-normalised embeddings (equivalent to cosine k-means), k selected by silhouette score, seed=42.
- **Cluster labelling:** TF-IDF ranking of distinctive n-grams in each cluster's title+abstract text.
- **Overlap definition:** SLR papers whose cluster (in the reporting landscape) is identified as a reporting-bias cluster by TF-IDF terms.

---

## Citation

If you use these materials, please cite:

> Vilchez Marin, R.S., & Velasquez Borasino, C.A. (2026). *Deep Learning and Spatiotemporal Modelling for Urban Crime Prediction: A Systematic Review of Architectures, Heterogeneous Data Fusion, and the Dark Figure Gap (2023–mid 2026)*. [journal TBD].
