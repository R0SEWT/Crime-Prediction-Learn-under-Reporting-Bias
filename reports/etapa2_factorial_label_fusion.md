# Factorial label × fusión, multi-seed (slr-muw / slr-2rry) — el veredicto

**Fecha:** 2026-07-13 (v4, 20-seed) · **Diseño:** {circular, honest(M=1), placebo}
× {concat, attention} × **20 seeds** (split distrital fijo). STGNN 150 épocas CPU,
eval Spearman intra-distrital macro vs oráculo geocodificado 2023. Contrastes
pareados por seed, CI95 bootstrap (B=5000). Datos crudos:
`data/silver/predictions/factorial_label_fusion{,_contrasts}.json`.

> **Cambio material vs v3 (3 seeds).** El factorial de 3 seeds **sobreestimó** el
> efecto de fusión ~2-3×. Con 20 seeds el 2×3 está completo (incluye la celda
> `placebo×concat` que faltaba) y los contrastes llevan CI pareado. Los 3 seeds
> originales (42/7/123) se reproducen exactos; el resto regresa las medias a la
> media. **La tesis central se refuerza; los claims positivos de fusión bajan.**

## Resultados (media ± sd de 20 seeds; nan-aware)

| celda | macro | secuestro | robo |
|:--|:--|:--|:--|
| circular × concat | 0.227 ± 0.049 | +0.120 ± 0.257 | 0.369 ± 0.004 |
| circular × attention | **0.256 ± 0.025** | +0.167 ± 0.049 | 0.381 ± 0.028 |
| honest × concat | 0.231 ± 0.051 | +0.135 ± 0.260 | 0.369 ± 0.003 |
| honest × attention | 0.250 ± 0.023 | +0.147 ± 0.060 | 0.377 ± 0.025 |
| placebo × concat | 0.230 ± 0.047 | +0.128 ± 0.248 | 0.369 ± 0.003 |
| placebo × attention | 0.254 ± 0.023 | +0.165 ± 0.050 | 0.379 ± 0.032 |

(placebo×concat secuestro: 19/20 seeds finitos — 1 seed colapsó la cabeza rara a
constante → ρ indefinido; se agrega nan-aware.)

## Contrastes pareados por seed (macro ρ, CI95)

**Fusión (attention − concat):**
- circular: **+0.029 [+0.006, +0.052] SIG**
- honest:   +0.019 [−0.009, +0.044] **ns**  ← v3 reportaba +0.074 "3-5 sd" (ruido de 3 seeds)
- placebo:  **+0.024 [+0.004, +0.045] SIG**  (celda nueva)
- **POOLED (60 pares): +0.024 [+0.010, +0.037] SIG**

**Label (bajo attention) — debe ser ~0:**
- honest − circular: −0.006 [−0.019, +0.005] ns
- honest − placebo:  −0.004 [−0.017, +0.009] ns
- placebo − circular: −0.002 [−0.014, +0.010] ns

**Interacción label×fusión:** (honest attn−concat) − (placebo attn−concat)
= −0.006 [−0.020, +0.009] **ns**.

## Veredicto por efecto (20 seeds)

1. **Intervención sobre el label: RESULTADO NEGATIVO, reforzado.** honest ≈
   circular ≈ placebo bajo attention (los tres contrastes cruzan 0, |Δ|≤0.006;
   interacción nula). Con λ_prior=0.5 y la loss de consistencia distrital —y
   dado que los tres labels comparten los MISMOS totales distritales por
   construcción, así que `L_district` es idéntico entre ellos— el STGNN es
   insensible al patrón intra-distrital del target. Es el hallazgo robusto.
2. **Fusión attention: efecto REAL pero PEQUEÑO.** +0.024 macro pooled (sig),
   no el +0.05–0.07 del estimado de 3 seeds. Significativo pooled y bajo
   circular/placebo; NO individualmente bajo honest. Sigue siendo el único
   efecto DL-positivo, pero su magnitud honesta es ~+0.02–0.03.
3. **Secuestro: es CONFIABILIDAD, no ganancia de media.** concat da ρ de
   secuestro con sd ~0.25 y **negativo en 9/20 seeds** (moneda al aire);
   attention lo deja **positivo en 20/20 seeds** (ρ ~0.15, sd ~0.05). El delta
   de media attn−concat NO es significativo (los CIs cruzan 0 porque la media
   de concat no es baja, solo inestable). El claim honesto: attention hace la
   cabeza rara *fiable*, no *más alta en promedio*.
4. **Techo DL ~0.25–0.26** (mejor celda circular×attention 0.256) — muy por
   debajo del prior histórico (0.394) y del techo tabular (0.464). La
   conclusión central se refuerza: el déficit DL no lo repara el label.

## Implicación para el manuscrito SIMBig (pendiente decisión de autor)

Abstract y §5.2 llevan los números de 3 seeds (+0.046/+0.074, "3-5 sd",
"rescata categorías raras"). Requieren:
- Fusión: +0.05–0.07 → **+0.024 pooled** (sig; ns bajo honest solo).
- Secuestro: "rescate" (media) → **fiabilidad** (0/20 vs 9/20 seeds negativos).
- Label null e interacción nula: se mantienen y se refuerzan (ahora 20 seeds,
  con la celda placebo×concat que cierra el 2×3).

## Reproducir

```bash
.venv-embed/bin/python scripts/exp_factorial_label_fusion.py --seeds 20
```
