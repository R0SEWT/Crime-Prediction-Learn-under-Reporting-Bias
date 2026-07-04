# Robustez temporal con test 2022 (slr-nvc)

**Protocolo:** se recalculan los contrastes con train 2019-2021 / test 2022 y se comparan con train 2019-2022 / test 2023. Modelo tabular HGB con los mismos parametros del ladder; deltas con bootstrap pareado sobre distritos (B=2000).

## Macros por modelo

| family       |   test_year | comparison     | metric    |   point |
|:-------------|------------:|:---------------|:----------|--------:|
| ladder       |        2022 | M1_demografia  | macro_rho |  0.37   |
| ladder       |        2022 | M1b_manzana    | macro_rho |  0.4418 |
| ladder       |        2022 | M4c_transporte | macro_rho |  0.4527 |
| ladder       |        2023 | M1_demografia  | macro_rho |  0.3798 |
| ladder       |        2023 | M1b_manzana    | macro_rho |  0.4547 |
| ladder       |        2023 | M4c_transporte | macro_rho |  0.4636 |
| transfer_aqp |        2022 | transfer       | macro_rho |  0.4355 |
| transfer_aqp |        2022 | prior_pob      | macro_rho |  0.4133 |
| transfer_aqp |        2022 | persistencia   | macro_rho |  0.4398 |
| transfer_aqp |        2023 | transfer       | macro_rho |  0.4855 |
| transfer_aqp |        2023 | prior_pob      | macro_rho |  0.4362 |
| transfer_aqp |        2023 | persistencia   | macro_rho |  0.5042 |

## Deltas inferenciales

| family       |   test_year | comparison                   |   point |   ci_low |   ci_high | significant   | replica   |
|:-------------|------------:|:-----------------------------|--------:|---------:|----------:|:--------------|:----------|
| ladder       |        2022 | M1b_manzana - M1_demografia  |  0.0719 |   0.0397 |    0.1004 | si            | si        |
| ladder       |        2022 | M4c_transporte - M1b_manzana |  0.0109 |  -0.0066 |    0.0312 | no            | si        |
| ladder       |        2023 | M1b_manzana - M1_demografia  |  0.0749 |   0.0478 |    0.1012 | si            | si        |
| ladder       |        2023 | M4c_transporte - M1b_manzana |  0.0089 |  -0.0094 |    0.0287 | no            | si        |
| transfer_aqp |        2022 | transfer - prior_pob         |  0.0222 |  -0.0164 |    0.0609 | no            | no        |
| transfer_aqp |        2022 | transfer - persistencia      |  0.0018 |  -0.0497 |    0.0552 | no            | si        |
| transfer_aqp |        2023 | transfer - prior_pob         |  0.0493 |   0.0065 |    0.0949 | si            | si        |
| transfer_aqp |        2023 | transfer - persistencia      | -0.01   |  -0.0733 |    0.0535 | no            | si        |

## Techo M4c

- M4c test 2022=0.4527; test 2023=0.4636; delta=-0.0109.

## Veredicto global

**REPLICA PARCIAL.** En test 2022 replican 3/4 contrastes inferenciales bajo los criterios preregistrados en este script.

Datos tabulares: `data/silver/predictions/temporal_robustness_2022.csv`.
