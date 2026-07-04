# Q3 — Generalizacion espacial

**Modelo:** M2_osm (mejor bloque Q4). **Train years:** 2019-2022. **Test year:** 2023.
**Data root:** `/home/rosewt-dell/Code/infelix`.

## Resultado principal

- Spatial grouped CV vs temporal holdout: Δrho=-0.103.
- Buffered CV vs grouped CV: Δrho=-0.051.
- Random H3 CV over buffered CV: +0.125 rho; esa brecha estima inflacion por leakage espacial.
- Lectura conservadora buffered: macro intra-rho=0.320 con std=0.158; la varianza por fold es alta.

## Resumen macro

| split               |   n_folds |   mean_macro_intra_rho |   std_macro_intra_rho |   mean_macro_intra_recall10 |   mean_macro_ndcg |   mean_train_districts |   mean_test_districts |
|:--------------------|----------:|-----------------------:|----------------------:|----------------------------:|------------------:|-----------------------:|----------------------:|
| temporal_holdout    |         1 |                  0.474 |                 0     |                       0.327 |             0.852 |                  nan   |                 nan   |
| random_h3_cv        |         5 |                  0.445 |                 0.044 |                       0.361 |             0.784 |                  nan   |                 nan   |
| spatial_grouped_cv  |         5 |                  0.371 |                 0.138 |                       0.284 |             0.765 |                   41.6 |                  10.4 |
| buffered_spatial_cv |         5 |                  0.32  |                 0.158 |                       0.256 |             0.71  |                   17   |                  10.4 |

## Veredicto Q3

La senal OSM generaliza parcialmente, pero no con la fuerza que sugiere el split temporal simple. La caida de temporal a grouped indica que parte del rendimiento depende de identidad distrital; la caida adicional a buffered indica leakage por vecindad. El claim defendible no es 'prediccion fina robusta en todo Lima', sino 'senal urbana OSM transferible con degradacion espacial sustantiva'.

Metricas completas: `data/silver/predictions/q3_spatial_cv_metrics.csv`.