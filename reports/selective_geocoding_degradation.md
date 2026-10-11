# Degradación selectiva de geocodificación: mecanismo + curva (slr-7lsf)

**Ancla:** limitación 4 del SIMBig — el binomial thinning uniforme (slr-dj7, `analysis/etapa2_geocoding_degradation.md`) es una cota optimista; aquí se caracteriza el mecanismo REAL de fallo de geocodificación en Lima y se re-corre la curva de evaluabilidad con pérdida selectiva empírica. Script: `scripts/eval_selective_degradation.py` (seed fija, regenerable).

Nota de base: la convención del experimento original usa 83.4%; en este universo la tasa CON COORDENADA medida es 84.7% y el éxito limpio (lo que sobrevive al anti-centroide) 46.8%.

## Fase 1 — mecanismo empírico de fallo

Universo: 1,144,718 denuncias limpias (solo_denuncia=1, 5 categorías, 2019-2023). Dos mecanismos de pérdida, reportados por separado:

- **(a) SIN COORDENADA**: 15.3% del universo.
- **(b) artefacto comisaría** (pile-up en coordenada exacta, umbral >30): 44.6% de las CON COORDENADA (432,326 denuncias en 41 distritos).
- **Éxito limpio** (lo que entra al oráculo, módulo filtro a celdas del matrix): 46.8%.

### (a) SIN COORDENADA — tasas por covariable

| crime_cat | n | tasa fail_nocoord | IC 95% |
|:--|--:|--:|:--|
| robo_hurto_callejero | 528,008 | 0.184 | [0.183, 0.185] |
| extorsion | 20,452 | 0.495 | [0.488, 0.502] |
| estafa | 43,926 | 0.328 | [0.323, 0.332] |
| violencia_familiar_sexual | 548,642 | 0.096 | [0.095, 0.097] |
| secuestro | 3,690 | 0.318 | [0.303, 0.333] |

| year | n | tasa fail_nocoord | IC 95% |
|:--|--:|--:|:--|
| 2019 | 228,185 | 0.292 | [0.290, 0.294] |
| 2020 | 175,282 | 0.097 | [0.096, 0.099] |
| 2021 | 198,721 | 0.100 | [0.099, 0.101] |
| 2022 | 230,831 | 0.122 | [0.121, 0.123] |
| 2023 | 311,699 | 0.141 | [0.139, 0.142] |

| virtual | n | tasa fail_nocoord | IC 95% |
|:--|--:|--:|:--|
| False | 1,105,727 | 0.159 | [0.158, 0.159] |
| True | 38,991 | 0.001 | [0.001, 0.002] |

| named_via | n | tasa fail_nocoord | IC 95% |
|:--|--:|--:|:--|
| False | 997,976 | 0.042 | [0.042, 0.042] |
| True | 146,742 | 0.911 | [0.910, 0.912] |

### (b) artefacto comisaría — tasas por covariable (entre las CON COORDENADA en bbox)

| crime_cat | n | tasa centroid | IC 95% |
|:--|--:|--:|:--|
| robo_hurto_callejero | 430,542 | 0.425 | [0.424, 0.427] |
| extorsion | 10,313 | 0.455 | [0.445, 0.464] |
| estafa | 29,499 | 0.405 | [0.400, 0.411] |
| violencia_familiar_sexual | 495,664 | 0.467 | [0.466, 0.468] |
| secuestro | 2,514 | 0.445 | [0.426, 0.465] |

| year | n | tasa centroid | IC 95% |
|:--|--:|--:|:--|
| 2019 | 161,439 | 0.426 | [0.424, 0.429] |
| 2020 | 158,119 | 0.506 | [0.503, 0.508] |
| 2021 | 178,765 | 0.492 | [0.490, 0.494] |
| 2022 | 202,518 | 0.425 | [0.423, 0.427] |
| 2023 | 267,691 | 0.409 | [0.407, 0.411] |

| virtual | n | tasa centroid | IC 95% |
|:--|--:|--:|:--|
| False | 929,601 | 0.453 | [0.452, 0.454] |
| True | 38,931 | 0.293 | [0.289, 0.298] |

### Gradiente distrital (distritos del feature matrix, n≥500) vs pobreza INEI 2018

- Spearman tasa SIN COORDENADA vs pobreza: **ρ=0.39** (p=1.2e-02, n=40)
- Spearman tasa artefacto comisaría vs pobreza: **ρ=0.64** (p=8.5e-06)
- Spearman tasa de éxito limpio vs pobreza: **ρ=-0.60** (p=4.8e-05)

| | éxito limpio | sin coord | artefacto | pobreza |
|:--|--:|--:|--:|--:|
| Santa Anita 150137 (peores 5) | 0.225 | 0.050 | 0.763 | 9.7% |
| Villa El Salvador 150142 (peores 5) | 0.329 | 0.228 | 0.573 | 20.1% |
| Puente Piedra 150125 (peores 5) | 0.344 | 0.196 | 0.572 | 22.7% |
| Pachacamac 150123 (peores 5) | 0.351 | 0.186 | 0.569 | 20.0% |
| Carabayllo 150106 (peores 5) | 0.354 | 0.212 | 0.550 | 19.2% |
| San Isidro 150131 (mejores 5) | 0.726 | 0.087 | 0.205 | 0.1% |
| Pueblo Libre 150121 (mejores 5) | 0.715 | 0.115 | 0.192 | 0.8% |
| San Luis 150134 (mejores 5) | 0.713 | 0.114 | 0.195 | 5.5% |
| Magdalena Del Mar 150120 (mejores 5) | 0.708 | 0.085 | 0.227 | 0.8% |
| San Borja 150130 (mejores 5) | 0.707 | 0.098 | 0.216 | 0.4% |

### Regresión logística — P(SIN COORDENADA)
(ref: robo_hurto_callejero, año 2019; SE Wald sin cluster por distrito — con n=1.1M los IC son optimistas, se leen como descriptivos)

| covariable | OR | IC 95% |
|:--|--:|:--|
| (intercepto) | 0.050 | [0.049, 0.052] |
| cat=extorsion | 4.694 | [4.487, 4.909] |
| cat=estafa | 2.531 | [2.442, 2.623] |
| cat=violencia_familiar_sexual | 0.676 | [0.663, 0.689] |
| cat=secuestro | 1.904 | [1.682, 2.156] |
| año=2020 | 1.300 | [1.261, 1.341] |
| año=2021 | 1.136 | [1.102, 1.171] |
| año=2022 | 1.037 | [1.008, 1.067] |
| año=2023 | 1.426 | [1.389, 1.464] |
| virtual | 0.027 | [0.021, 0.035] |
| named_via | 235.780 | [230.418, 241.267] |
| pobreza_z | 1.415 | [1.397, 1.433] |

### Regresión logística — P(artefacto comisaría | CON COORDENADA en bbox)

| covariable | OR | IC 95% |
|:--|--:|:--|
| (intercepto) | 0.852 | [0.842, 0.862] |
| cat=extorsion | 1.081 | [1.039, 1.125] |
| cat=estafa | 0.915 | [0.893, 0.938] |
| cat=violencia_familiar_sexual | 1.058 | [1.049, 1.067] |
| cat=secuestro | 1.008 | [0.931, 1.091] |
| año=2020 | 1.368 | [1.349, 1.387] |
| año=2021 | 1.308 | [1.291, 1.326] |
| año=2022 | 0.996 | [0.983, 1.010] |
| año=2023 | 1.015 | [1.002, 1.028] |
| virtual | 0.554 | [0.541, 0.568] |
| pobreza_z | 1.308 | [1.300, 1.315] |

## Fase 2 — curva de evaluabilidad: selectivo vs uniforme

Mismo protocolo que slr-dj7 (M4c re-entrenado sobre targets degradados, 10 seeds por nivel, base 83.4%); 'selectivo' = retención por celda ∝ propensión de éxito limpio de su distrito×categoría (shrinkage m=50), escalada por nivel; 'cobertura' = nº medio de distritos evaluables por categoría en el rho medible.

| tasa geocod. | mecanismo | ρ medible (±sd) | ρ real (±sd) | ρ persistencia | cobertura (distritos) | % seeds feat>persist |
|--:|:--|:--|:--|--:|--:|--:|
| 83.4% | uniforme | 0.464 (±0.000) | 0.464 (±0.000) | 0.401 | 39.0 | 100% |
| 83.4% | selectivo | 0.464 (±0.000) | 0.464 (±0.000) | 0.401 | 39.0 | 100% |
| 80.0% | uniforme | 0.457 (±0.006) | 0.460 (±0.006) | 0.397 | 39.0 | 100% |
| 80.0% | selectivo | 0.437 (±0.007) | 0.452 (±0.009) | 0.378 | 38.8 | 100% |
| 60.0% | uniforme | 0.441 (±0.010) | 0.458 (±0.007) | 0.376 | 38.5 | 100% |
| 60.0% | selectivo | 0.406 (±0.012) | 0.445 (±0.008) | 0.342 | 37.8 | 100% |
| 50.0% | uniforme | 0.432 (±0.007) | 0.458 (±0.005) | 0.360 | 38.2 | 100% |
| 50.0% | selectivo | 0.395 (±0.009) | 0.447 (±0.008) | 0.332 | 37.7 | 100% |
| 40.0% | uniforme | 0.413 (±0.011) | 0.454 (±0.007) | 0.339 | 37.8 | 100% |
| 40.0% | selectivo | 0.388 (±0.007) | 0.441 (±0.006) | 0.320 | 36.9 | 100% |
| 24.9% | uniforme | 0.386 (±0.015) | 0.447 (±0.010) | 0.310 | 36.5 | 100% |
| 24.9% | selectivo | 0.354 (±0.011) | 0.435 (±0.009) | 0.270 | 35.7 | 100% |
| 10.0% | uniforme | 0.324 (±0.011) | 0.429 (±0.013) | 0.216 | 34.0 | 100% |
| 10.0% | selectivo | 0.293 (±0.016) | 0.415 (±0.013) | 0.223 | 32.9 | 100% |

### Brecha selectivo − uniforme (Δρ; negativo = el thinning uniforme era optimista)

| tasa geocod. | Δ medible | Δ real | Δ persistencia |
|--:|--:|--:|--:|
| 83.4% | +0.000 | +0.000 | +0.000 |
| 80.0% | -0.020 | -0.008 | -0.019 |
| 60.0% | -0.035 | -0.013 | -0.034 |
| 50.0% | -0.037 | -0.011 | -0.028 |
| 40.0% | -0.024 | -0.013 | -0.019 |
| 24.9% | -0.032 | -0.013 | -0.041 |
| 10.0% | -0.031 | -0.013 | +0.007 |

