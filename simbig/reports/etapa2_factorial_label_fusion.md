# Factorial label × fusión, multi-seed (slr-muw) — el veredicto

**Fecha:** 2026-07-03 · **Diseño:** {circular, híbrido-M1} × {concat, attention}
× 3 seeds (42/7/123; split distrital fijo) + placebo(attention) × 3.
STGNN 150 épocas CPU, eval Spearman intra-distrital macro vs oráculo
geocodificado 2023. Datos crudos: `data/silver/predictions/
{label_dose_response_multiseed,factorial_label_fusion}.json`.

## Resultados (media ± sd de 3 seeds)

| celda | macro | secuestro | robo |
|:--|:--|:--|:--|
| circular × concat | 0.201 ± 0.017 | −0.029 ± 0.068 | 0.367 ± 0.001 |
| circular × attention | 0.247 ± 0.013 | +0.107 ± 0.038 | 0.381 ± 0.026 |
| M1 × concat | 0.190 ± 0.011 | −0.094 ± 0.046 | 0.369 ± 0.002 |
| M1 × attention | **0.264 ± 0.013** | **+0.165 ± 0.044** | 0.387 ± 0.007 |
| placebo × attention | 0.244 ± 0.029 | +0.140 ± 0.075 | 0.382 ± 0.030 |

Barrido M complementario (attention): M=10 0.252±0.028, M=50 0.253±0.021.

## Veredicto por efecto

1. **Fusión attention: EFECTO REAL.** +0.046 sobre concat con label circular y
   +0.074 con label M1 (~3-5 sd). Es el único efecto DL-positivo que sobrevive
   el rigor multi-seed.
2. **El "rescate de secuestro" es 100% de la atención, no del label.** Concat
   produce ρ negativo en secuestro con cualquier label; attention lo vuelve
   positivo con cualquier label — *incluido el placebo de patrón permutado*.
   La fusión cross-modal estabiliza las cabezas de categorías raras vía la
   representación, independientemente del target.
3. **Intervención sobre el label: RESULTADO NEGATIVO.** M1 ≈ circular ≈
   placebo dentro de ruido (Δ máx +0.017, ~1.3 sd). El STGNN es insensible al
   patrón intra-distrital de su target: con λ_prior = 0.5 y la loss de
   consistencia distrital, el gradiente del patrón fino apenas influye. La
   honestidad del label es necesaria para *interpretar* (sin ella, ρ alto vs
   target = circularidad) pero no suficiente para *mejorar* esta arquitectura.
4. **El techo DL queda en ~0.26 ± 0.03** en la mejor configuración — muy por
   debajo del prior histórico (0.394) y del techo tabular (0.464). La
   conclusión central se refuerza: el déficit del DL no es reparable por el
   label; es de arquitectura/régimen de datos.

## Corrección al registro

La tabla single-seed de `etapa2_hybrid_target.md` (circular-concat 0.197 →
híbrido-attention 0.276, leído como "sinergia label×fusión") queda SUPERSEDIDA:
el contraste mezclaba hardware (Lightning vs local) y seeds individuales. El
componente label de esa sinergia era ruido; el componente fusión era real.
Los claims derivados en ch5 §Q2b y design/columna-vertebral se corrigen en
este mismo commit.

## Reproducir

```bash
.venv-embed/bin/python scripts/exp_dose_response_multiseed.py
.venv-embed/bin/python scripts/exp_factorial_label_fusion.py
```
