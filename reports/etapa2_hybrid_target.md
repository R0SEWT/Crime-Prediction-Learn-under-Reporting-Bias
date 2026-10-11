# Target híbrido H3 (slr-us4): patrón geocodificado × escala de-sesgada

> ⚠️ **REVISADO por multi-seed (2026-07-03, slr-muw):** la tabla de abajo es
> single-seed y el contraste "+0.079 sinérgico" NO sobrevive el rigor
> multi-seed: con 3 seeds las variantes de label son indistinguibles bajo
> atención (circular 0.247±0.013, M=1 0.264±0.013, M=10 0.252±0.028,
> placebo 0.244±0.029). El 0.2079 circular provenía de un run en Lightning
> (hardware distinto) y el 0.2762 híbrido de una seed favorable. Lectura
> vigente: el STGNN es INSENSIBLE al patrón intra-distrital del label — la
> honestidad del label es necesaria pero no suficiente; ver
> `label_dose_response_multiseed.json` y el factorial label×fusión.

**Fecha:** 2026-07-03 · **Hardware:** CPU local (`.venv-embed`, torch 2.12) · **Épocas:** 150 · seed 42

## Motivación

El target latente H3 original reparte el latente distrital por población
(`build_latent_surface_h3.py`, patrón `population`): la única variación
intra-distrital del label es población → todo modelo entrenado sobre él aprende
el prior poblacional (circular; ver `etapa2_q1_observado_vs_latente.md` y el
gate Q2). El target **híbrido** rompe esa circularidad:

```
w_h3 = (count_geo_h3 + M · popshare_h3) / (count_geo_distrito + M)      M = 10
latente_h3 = latente_distrito × w_h3          (por categoría × año)
```

- Patrón intra-distrital = denuncias **geocodificadas reales** (`h3_observed_geocoded`,
  ~31% de coords reales), con suavizado Dirichlet hacia el prior poblacional
  (celdas/distritos sin geocodificación caen suavemente a popshare).
- Escala = latente distrital **de-sesgado** de Etapa 1 (conservación distrital
  verificada: error relativo máx 1e-7).
- Join incluye `ubigeo` (celdas fronterizas aparecen bajo dos distritos en el
  oráculo; sin eso el merge duplica filas — bug corregido).

Entrenamiento con split honesto ya existente: años 2019-2022 train / 2023 test,
25% de distritos held-out. El patrón geocodificado 2023 **no** entra al train.

## Resultado (Spearman intra-distrital vs oráculo, 2023, macro por categoría)

| categoría                  | concat-circular | hyb-concat | **hyb-attention** | hyb-attn λ₀ |
|:---------------------------|----------------:|-----------:|------------------:|------------:|
| robo_hurto_callejero       |          0.3683 |     0.3669 |        **0.3930** |      0.3488 |
| extorsión                  |          0.1301 |     0.1240 |        **0.1978** |      0.1191 |
| estafa                     |          0.2123 |     0.2110 |          0.2253   |  **0.2358** |
| violencia_familiar_sexual  |          0.3367 |     0.3427 |        **0.3509** |      0.3348 |
| secuestro                  |         −0.0608 |     0.0418 |        **0.2142** |      0.1422 |
| **MACRO**                  |          0.1973 |     0.2173 |        **0.2762** |      0.2362 |

Referencias: attention-circular (Lightning, T4) 0.2079 · prior histórico 0.394 ·
techo ladder tabular M4c 0.4636.

## Lectura

1. **El label era el cuello de botella, y arreglarlo es sinérgico con la fusión.**
   Cada fix por separado aporta poco (circular→híbrido con concat: +0.020;
   concat→attention con target circular: +0.011), pero juntos: **0.197 → 0.276
   (+0.079)**, con mejora en las **cinco** categorías y secuestro pasando de
   ρ negativo a 0.214. La fusión cross-modal solo puede explotar señal que el
   label contiene.

2. **λ_prior=0.5 ayuda incluso con label honesto** (0.2762 vs 0.2362 sin él):
   la loss de consistencia con el prior actúa como regularizador contra el
   overfitting del patrón geocodificado sparse, no como lastre circular.

3. **El techo sigue sin romperse.** El mejor DL (0.2762) queda bajo el prior
   histórico (0.394) y muy bajo el techo tabular (0.4636). La brecha restante
   ya no es el label: es generalización espacial + el hecho de que XGB por
   categoría explota mejor las features tabulares finas que un tronco GCN
   multi-tarea. Nota: para la métrica intra-distrital, entrenar XGB sobre el
   híbrido ≈ entrenarlo sobre el oráculo suavizado (la escala distrital es
   constante intra-distrito y no altera rankings) — el ladder tabular ES esa
   referencia.

4. **Implicación para ch5:** el argumento "el DL aprende el prior porque el
   target es circular" queda demostrado por intervención (arreglar el target
   mueve +0.08 macro), pero la conclusión central no cambia: la contribución
   está en el de-sesgo (Etapa 1) + features finas; la arquitectura DL con
   fusión real es defendible como *demostración multimodal honesta*, no como
   ganador predictivo.

## Reproducir

```bash
python3 scripts/build_latent_surface_h3.py --pattern geocoded   # --smooth 10
.venv-embed/bin/python scripts/train_stgnn.py --target hybrid --epochs 150 --no-mlflow
.venv-embed/bin/python scripts/train_stgnn.py --target hybrid --fusion attention --epochs 150 --no-mlflow
python3 scripts/eval_fusion_oracle.py
```

Artefactos (gitignored): `data/silver/crime_latent_surface_hybrid.parquet`,
`data/silver/predictions/stgnn_latente_h3_hybrid{,_attention}.parquet`.
Reportes de entrenamiento: `analysis/stgnn_hybrid_{concat,attention}_metrics.md`.
