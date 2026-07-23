# Companion paper — replication package

**Paper:** *Estimating Latent Urban Crime Risk under Reporting Bias: A Bayesian Small Area
Approach for Lima Metropolitana* (preprint; Vilchez Marin & Velasquez Borasino, UPC).

This directory is the public replication repository referenced in the paper's
*Availability of data and materials*. It contains the analysis scripts, the taxonomy
crosswalk, figure-reproducibility code, and the derived (district-level, aggregated)
data artefacts behind every load-bearing number in the paper. No survey microdata is
redistributed: ENAPRES microdata is publicly available from INEI
(`https://iinei.inei.gob.pe/microdatos/`), MININTER denuncias from
`seguridadciudadana.mininter.gob.pe`, and SINADEF from the national open-data portal.

## The anti-stale mechanism (read this first)

Every load-bearing figure in the paper (multipliers, reporting rates, correlations, CIs,
bounds) is **emitted by a script into `registry/canonical_numbers.json`** and the paper
cites it through `CANON:` anchors validated by `scripts/check_canon.py`. If you recompute
an artefact, the registry records the emitting script's hash; a stale citation fails the
check. This is how the paper guarantees its prose matches its pipeline.

## Manifest — artefacts → paper sections

| file | feeds | contents |
|---|---|---|
| `registry/canonical_numbers.json` | all | canonical values, policies, provenance (script, commit, input hashes) |
| `artifacts/latente_distrito_categoria_anio.csv` | §5.1–§5.4 | observed vs latent counts per district × category × year (43 Lima districts) |
| `artifacts/composition_excl_dv.json` | §5.4, Table 1 | 4-category composition shares and L1 distances (annualised + raw ENAPRES bases) |
| `artifacts/latent_ci.json` | §4.5 | per-district delta-method SEs and Monte Carlo multiplier CIs, all categories |
| `artifacts/maup_sensitivity.json`, `artifacts/maup.md` | §6.2 | re-zoning diagnostics: 6 scales × 200 random partitions |
| `artifacts/fraud_bounds.json` | §6.3, §8 | Kreider–Pepper partial-identification regions (ν grid), Lima + national |
| `artifacts/district_trust_pnp.json` | §5.3 | district trust (ENAPRES P616A_1) vs robbery multiplier |
| `artifacts/sinadef_validation.md` | §5.5 | external homicide-registry anchor, rate-vs-rate (null result, documented) |

## Scripts — pipeline → paper

| script | role |
|---|---|
| `harmonize_taxonomy.py` | MININTER ↔ ENAPRES category crosswalk (§4.2), version-controlled |
| `build_reporting_rate.py` | victim/incident reporting rates: direct + EB Beta-Binomial (§4.4) |
| `build_latent_surface.py` | latent surface λ* = y/r̂ (§4.1) |
| `compute_latent_ci.py` | §4.5 uncertainty: per-district delta SEs + MC multiplier CIs |
| `recheck_phi_cap.py`, `phi_lrt_bootstrap.py` | §5.3 heterogeneity: φ profile, pseudo-LRT, bootstrap calibration, design-effect sensitivity |
| `build_composition_canon.py` | §5.4 composition shares and L1 (canonical generator) |
| `validate_latent.py`, `validate_sinadef_convergence.py` | §5.5 convergent validity + SINADEF anchor |
| `build_district_trust.py` | §5.3 trust × multiplier |
| `maup_sensitivity.py` | §6.2 re-zoning |
| `sensitivity_definition.py` | §6.1 robbery-definition trio (completed-only / attempt-inclusive / 2024-only) |
| `eval_fraud_bounds.py` | §6.3 Kreider–Pepper bounds |
| `eval_covid_robustness_rates.py` | §3.1 non-pandemic robustness of r̂ |
| `sparsity_diagnostic.py` | §4.3 estimator-selection diagnostic |
| `figures_desesgo.py` | all paper figures |
| `canon.py`, `check_canon.py` | canonical-numbers registry + validator |

## Regeneration

Scripts expect the repository layout of the main (private) research repo — paths under
`data/datasets/enapres/raw/*.zip` (ENAPRES CAP_600 waves 2019–2024, downloadable from
INEI) and the derived silver artefacts. With those in place:

```bash
python3 build_reporting_rate.py --years 2019-2024        # rates (victim + incident)
python3 build_latent_surface.py                          # latent surface
python3 validate_latent.py && python3 build_composition_canon.py
python3 compute_latent_ci.py                             # §4.5 CIs (200k draws, seeded)
python3 maup_sensitivity.py --reps 200                   # §6.2
python3 eval_fraud_bounds.py                             # §6.3
python3 check_canon.py --all                             # verify nothing is stale
```

All stochastic steps use fixed seeds; the surface build is byte-deterministic given
identical inputs.
