# What Survives an Uneven Record? Fine-Grained Crime Prediction under Reporting Bias: Evidence from Peru

**Authors:** Rody Sebastian Vilchez Marin · Christian Aaron Velasquez Borasino ·
Carlos Adrian Alarcon Delgado (Universidad Peruana de Ciencias Aplicadas — UPC, Lima, Peru)

**Venue:** SIMBig 2026, Springer CCIS (to appear)

This repository is the replication package cited in the paper. Every table and
figure in the manuscript maps to a script (protocol), a research report
(adjudicated reading, written before the manuscript), and, where small enough
to version, the metrics file the numbers come from.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | experiment, evaluation and figure scripts |
| `reports/` | research reports behind each manuscript item |
| `metrics/` | small metrics files behind published numbers |
| `registry/canonical_numbers.json` | canonical values for the figures the manuscript anchors (see *Registry*) |
| `requirements.txt` | Python dependencies of the deposited scripts |

## Manuscript → evidence map

| Manuscript item | Script(s) | Report(s) | Metrics |
|---|---|---|---|
| Table 1 — geocoding rates by city | `scripts/transfer_arequipa.py` (feasibility pass) | `reports/etapa2_transfer_feasibility.md` | — |
| Table 2 — estimator settings (GBM and STGNN hyperparameters) | `scripts/train_stgnn.py` (argparse defaults = protocol), `scripts/exp_factorial_label_fusion.py` (seeds, M=1 hybrid), `scripts/eval_modality_ladder.py` (GBM) | `reports/etapa2_epochs_ablation.md` (sensitivity: 600 epochs, λ_cell) | — |
| Table 3 — source-attribution ladder, with the family-wise correction (Westfall–Young step-down, simultaneous critical value 2.35<!-- CANON: ladder_mult.simultaneous_crit = 2.35 -->, adjusted p-values) | `scripts/eval_modality_ladder.py`, `scripts/eval_metric_robustness.py`, `scripts/exp_ladder_multiplicity.py` (family-wise correction) | `reports/etapa2_q4_modalidad_ladder.md`, `reports/etapa2_tabular_ceiling.md`, `reports/etapa2_metric_robustness.md`, `reports/etapa2_ladder_multiplicity.md` | `metrics/q4_modality_ladder_metrics.csv` |
| Table 4 — factorial label × fusion (+ placebo), M=1 hybrid | `scripts/exp_factorial_label_fusion.py`, `scripts/exp_dose_response_multiseed.py`, `scripts/train_stgnn.py`, `scripts/build_latent_surface_h3.py`, `scripts/eval_fusion_oracle.py` (concat vs attention scored against the geocoded reference), `scripts/eval_observed_vs_latent.py` (observed vs latent surface against the same reference) | `reports/etapa2_factorial_label_fusion.md`, `reports/etapa2_hybrid_target.md`, `reports/etapa2_q1_observado_vs_latente.md` | `metrics/factorial_label_fusion.json`, `metrics/label_dose_response.json`, `metrics/label_dose_response_multiseed.json`, `metrics/q1_observed_vs_latent_metrics.csv` |
| Table 5 — three-city transfer | `scripts/transfer_arequipa.py` (`--city arequipa\|piura\|cusco`) | `reports/etapa2_transfer_arequipa.md`, `reports/etapa2_transfer_piura.md`, `reports/etapa2_transfer_cusco.md` | — |
| Table 6 — geocoding degradation, uniform (MCAR bound) and selective (empirical propensity) | `scripts/eval_selective_degradation.py` (one run supplies both arms; supersedes `scripts/exp_geocoding_degradation.py`, kept for the submitted-version numbers) | `reports/selective_geocoding_degradation.md`, `reports/etapa2_geocoding_degradation.md` | `metrics/selective_geocoding_degradation.csv`, `metrics/geocoding_degradation.csv` |
| Figure 2 — learning curve (paired deltas) | `scripts/exp_learning_curve.py`, `scripts/figures_simbig.py` | `reports/etapa2_learning_curve.md` | — |
| Figure 3 — evaluability curve (uniform + selective) | `scripts/eval_selective_degradation.py`, `scripts/figures_simbig.py` | `reports/selective_geocoding_degradation.md` | `metrics/selective_geocoding_degradation.csv` |
| §2.1 — evaluation regime of the DL corpus (how prior work evaluates: observed counts, spatial splits, paired uncertainty) | — | `reports/etapa2_eval_regime_corpus.md` | — |
| §4.3 — random-CV inflation (+0.125 / +0.103) | `scripts/eval_spatial_cv.py` | `reports/etapa2_q3_spatial_cv.md` | — |
| §5.2 / Limitations — epochs & λ ablation (M=10 cell, three seeds) | `scripts/exp_epochs_ablation.py` | `reports/etapa2_epochs_ablation.md` | — |
| §5.4 / Ethics — poverty gradient of geocoding failure (ρ = 0.64<!-- CANON: geo_failure.rho_centroid_pov = 0.64 -->, −0.60<!-- CANON: geo_failure.rho_clean_pov = -0.60 -->) | `scripts/eval_selective_degradation.py` (phase 1) | `reports/selective_geocoding_degradation.md` | — |
| Limitations — attention interpretability (negative result) | `scripts/exp_attention_weights.py` | `reports/etapa2_attention_interpretability.md` | `metrics/attention_weights_summary.csv` |
| Limitations — test-2022 year robustness | `scripts/exp_temporal_robustness_2022.py` | `reports/etapa2_temporal_robustness.md` | `metrics/temporal_robustness_2022.csv` |
| Contribution 2 scoping ("first placebo label intervention") | — | `reports/antifalsify_c2.md` (near-miss adjudication), `reports/antifalsify_l3.md` (1,779-record venue-agnostic screen) | — |

**Shared modules** (imported by the scripts above, not run on their own):
`scripts/canon.py` (canonical-numbers registry: emission and provenance),
`scripts/etapa2_metrics.py` (rank, top-k and error metrics, intra-district
evaluation and bootstrap intervals), `scripts/exp_label_dose_response.py` (hybrid-label construction and
placebo, used by the factorial and the multi-seed dose-response),
`scripts/experiment_tracking.py` (optional MLflow logging for
`scripts/train_stgnn.py`), `scripts/research_paths.py` (raw-data locations, see
*Data*), `scripts/build_observed_h3_points.py` (category mapping of the geocoded
reference), and `scripts/fetch_inei_manzana_h3.py`, `scripts/fetch_osm_h3.py`,
`scripts/fetch_osm_proximity_h3.py` (feature builders that
`scripts/transfer_arequipa.py` reuses for the transfer cities).

Numbering follows the camera-ready manuscript (Figure 1 is the study-design
diagram). The submitted version (August 2026) reported Table 6 from
`scripts/exp_geocoding_degradation.py` (uniform thinning only); the camera-ready
re-ran both arms in one script so table and figure share seeds, which moves the
uniform values by ≤0.008. The camera-ready also corrects the hybrid-label
smoothing stated in §5.2: the 20-seed factorial used M=1 (not M=10).

## Data

The complaint registry is open data from Peru's Interior Ministry
(SIDPOL/MININTER, *Datos Abiertos*); all feature sources are open (INEI, OSM,
WorldPop, SRTM, ESA WorldCover, Sentinel-2, VIIRS, Meta HRSL, COFOPRI, CENAMA,
ESCALE, MINSA). The H3 feature matrices, label surfaces and prediction parquets
are large and regenerable from those sources, so they are not versioned here;
the metrics behind every published number are.

Scripts expect the directory layout of the authors' research repository
(`scripts/`, `analysis/`, `data/silver/...`); paths inside them are relative
to that root. The raw complaint exports are read from environment variables,
with a repository-relative default when unset:

| Input | Variable | Default |
|---|---|---|
| Lima complaint dump (SIDPOL/MININTER export, one parquet) | `INFELIX_LIMA_PARQUET` (also `INFELIX_LIMA_DENUNCIAS`) | `data/raw/external/LIMA.parquet` |
| National complaint corpus (directory of parquet files, transfer cities) | `INFELIX_NATIONAL_CORPUS` | `data/raw/external/data_denuncias` |

## Environment

`pip install -r requirements.txt` installs everything the deposited scripts
import, grouped in the file as:

- **Tabular experiments and evaluation:** numpy, pandas, pyarrow, scipy,
  scikit-learn and h3.
- **Transfer-city features:** geopandas, shapely, pyogrio, rasterio, requests
  and tqdm.
- **Neural experiments** (`scripts/train_stgnn.py`, the factorial, the
  dose-response and the epochs ablation): torch, torch-geometric and lightning.
  MLflow is optional: the protocol scripts pass
  `--no-mlflow`.
- **Figures:** matplotlib, geopandas and h3.
- **Determinism:** all model fits use fixed seeds (42 unless multi-seed by
  design: {42, 7, 123}); bootstrap uses B=2000 with a seeded RNG; the paired
  bootstrap resamples districts, the replication unit of the intra-district
  metric.

The reports under `reports/` are research artifacts written **before** the
manuscript prose, including negative results (attention interpretability
NO-GO, label-intervention null) and one in-place superseded reading
(`reports/etapa2_hybrid_target.md` header): they document what was corrected,
not only what survived.

## Registry

`registry/canonical_numbers.json` holds the canonical value of every
load-bearing figure that the manuscript anchors (geocoding-failure rates and
their poverty gradient, the degradation results, the source-attribution ladder
multiplicity correction, and the two district-level dark-figure multipliers
the paper quotes). It is projected from the authors' full registry to exactly
the families the manuscript cites; nothing else is included.

Each entry records its display value, estimator, unit and provenance: the
emitting script, the SHA-256 of that script and of its inputs, the git commit
and the emission timestamp. Input paths are relative to the authors' research
repository, except external raw inputs, which appear as `external/<file>`.
`dirty: true` means the emitting script had uncommitted edits at emission time;
`script_sha256` is then the integrity reference for the exact script that
produced the value. The `multiplier.*` entries originate in the companion
paper's pipeline (Vilchez Marin & Velasquez Borasino, *Estimating Latent Urban
Crime Risk under Reporting Bias: A Bayesian Small Area Approach for Lima
Metropolitana*, 2026, manuscript in preparation; emitted by its
build_latent_surface script, which is not deposited here) and are cited as such
in the manuscript.

## Citation

> Vilchez Marin, R.S., Velasquez Borasino, C.A., Alarcon Delgado, C.A. (2026).
> What Survives an Uneven Record? Fine-Grained Crime Prediction under Reporting
> Bias: Evidence from Peru. In: SIMBig 2026, Communications in Computer and
> Information Science. Springer (to appear).

Archived snapshot: DOI pending.

## License

License: to be added.
