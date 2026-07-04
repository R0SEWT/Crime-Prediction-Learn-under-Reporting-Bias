# SIMBig 2026 — Experimental package

**Paper:** *What Can Fine-Grained Crime Prediction Learn under Reporting Bias?
Label Circularity, Cross-Modal Fusion, and Cross-City Transfer in Peru*

**Authors:** Christian Aaron Velasquez Borasino · Rody Sebastian Vilchez Marin
(Universidad Peruana de Ciencias Aplicadas — UPC, Lima, Peru)

Every table and figure in the manuscript maps to a script (protocol), a
research report (adjudicated reading, written before the manuscript), and —
where small enough to version — the metrics file the numbers come from.

## Manuscript → evidence map

| Manuscript item | Script(s) | Report(s) | Metrics |
|---|---|---|---|
| Table 1 — evaluation regime of the DL corpus | (extraction from review corpus, full-text verified) | `reports/etapa2_eval_regime_corpus.md` | — |
| Table 2 — geocoding rates by city | `scripts/transfer_arequipa.py` (feasibility pass) | `reports/etapa2_transfer_feasibility.md` | — |
| Table 3 — source-attribution ladder | `scripts/eval_modality_ladder.py`, `scripts/eval_metric_robustness.py` | `reports/etapa2_q4_modalidad_ladder.md`, `etapa2_tabular_ceiling.md`, `etapa2_metric_robustness.md` | `metrics/q4_modality_ladder_metrics.csv` |
| Table 4 — factorial label × fusion (+ placebo) | `scripts/exp_factorial_label_fusion.py`, `exp_dose_response_multiseed.py`, `train_stgnn.py`, `build_latent_surface_h3.py` | `reports/etapa2_factorial_label_fusion.md`, `etapa2_hybrid_target.md`, `etapa2_q1_observado_vs_latente.md` | `metrics/factorial_label_fusion.json`, `label_dose_response*.json` |
| Table 5 — three-city transfer | `scripts/transfer_arequipa.py` (`--city arequipa\|piura\|cusco`) | `reports/etapa2_transfer_{arequipa,piura,cusco}.md` | — |
| Table 6 — geocoding degradation | `scripts/exp_geocoding_degradation.py` | `reports/etapa2_geocoding_degradation.md` | `metrics/geocoding_degradation.csv` |
| Figure 1 — oracle vs prediction maps | `scripts/figures_simbig.py` | — (reuses ladder/transfer protocols) | — |
| Figure 2 — learning curve (paired deltas) | `scripts/exp_learning_curve.py`, `figures_simbig.py` | `reports/etapa2_learning_curve.md` | — |
| Figure 3 — evaluability curve | `scripts/exp_geocoding_degradation.py`, `figures_simbig.py` | `reports/etapa2_geocoding_degradation.md` | `metrics/geocoding_degradation.csv` |
| §2.3 — random-CV inflation (+0.125 / +0.103) | `scripts/eval_spatial_cv.py` | `reports/etapa2_q3_spatial_cv.md` | — |
| §5.2 / Limitations — epochs & λ ablation | `scripts/exp_epochs_ablation.py` | `reports/etapa2_epochs_ablation.md` | — |
| Limitations — attention interpretability (negative result) | `scripts/exp_attention_weights.py` | `reports/etapa2_attention_interpretability.md` | `metrics/attention_weights_summary.csv` |
| Limitations — test-2022 year robustness | `scripts/exp_temporal_robustness_2022.py` | `reports/etapa2_temporal_robustness.md` | `metrics/temporal_robustness_2022.csv` |
| Contribution 2 scoping ("first placebo label intervention") | — | `reports/antifalsify_c2.md` (near-miss adjudication), `antifalsify_l3.md` (1,779-record venue-agnostic screen) | — |

## Reproducibility notes

- **Data.** The complaint registry is open data from Peru's Interior Ministry
  (SIDPOL/MININTER, *Datos Abiertos*); all feature sources are open (INEI,
  OSM, WorldPop, SRTM, ESA WorldCover, Sentinel-2, VIIRS, Meta HRSL, COFOPRI,
  CENAMA, ESCALE, MINSA). Intermediate parquets are large and regenerable, so
  they are not versioned here; the metrics behind every published number are.
- **Environment.** Tabular experiments: Python 3 + numpy/scipy/pandas/
  scikit-learn. Neural experiments (`train_stgnn.py`, factorial, epochs):
  torch/transformers per `../code/requirements-embed.txt`. Figures:
  matplotlib + geopandas + h3.
- **Determinism.** All model fits use fixed seeds (42 unless multi-seed by
  design: {42, 7, 123}); bootstrap uses B=2000 with seeded RNG; the paired
  bootstrap resamples districts, the replication unit of the intra-district
  metric.
- The reports under `reports/` are research artifacts written **before** the
  manuscript prose, including negative results (attention interpretability
  NO-GO, label-intervention null) and one in-place superseded reading
  (`etapa2_hybrid_target.md` header): they document what was corrected, not
  only what survived.
