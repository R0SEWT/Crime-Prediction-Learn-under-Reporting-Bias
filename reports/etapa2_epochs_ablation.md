# Ablación épocas/lambda (slr-p1u.10)

Celda híbrido(M=10)×attention. Referencias 150ep/lambda=0.5: 0.252±0.028 (multiseed).

| config | macro rho ± sd | seeds |
|:--|:--|:--|
| 600ep_lam0.5 | 0.1907 ± 0.0312 | [0.1889, 0.1605, 0.2228] |
| 150ep_lam0.1 | 0.2373 ± 0.0162 | [0.223, 0.234, 0.2548] |
| 150ep_lam1.0 | 0.2294 ± 0.0241 | [0.2572, 0.2171, 0.214] |

## Veredicto

El techo DL NO es artefacto de presupuesto de entrenamiento — es peor que eso
para la arquitectura: **entrenar 4× más EMPEORA el resultado** (600ep:
0.191±0.031 vs 150ep: 0.252±0.028), aun con selección de best-checkpoint por
validación. Optimizar el label más duro aleja al modelo del oráculo, porque el
label no contiene la señal intra-distrital del oráculo (consistente con el
factorial: insensibilidad al patrón del label). λ∈{0.1, 1.0} queda dentro/bajo
la banda (0.237/0.229): tampoco es la regularización. El ~0.25-0.26 de 150
épocas es una ENVOLVENTE SUPERIOR: más cómputo mueve hacia abajo, no hacia
arriba. Respuesta cerrada a la objeción de underfitting.
