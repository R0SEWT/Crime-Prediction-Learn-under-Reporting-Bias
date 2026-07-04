# Transferencia espacial Lima → Arequipa (slr-211)

**Protocolo:** modelo HGB (params del ladder) con el bloque tabular TRANSFERIBLE (18 features: WorldPop, manzana censal INEI, OSM sin features Lima-específicas). Train Lima 2019-2022 → test **Arequipa 2023** vs oráculo geocodificado propio. Spearman intra-distrital macro (distritos con ≥5 celdas).

| categoría | transfer (Lima→AQP) | local (AQP→AQP) | persistencia | prior pob. |
|:--|--:|--:|--:|--:|
| robo_hurto_callejero | 0.548 | 0.693 | 0.583 | 0.541 |
| extorsion | 0.408 | 0.458 | 0.489 | 0.395 |
| estafa | 0.530 | 0.602 | 0.559 | 0.500 |
| violencia_familiar_sexual | 0.515 | 0.656 | 0.668 | 0.397 |
| secuestro | 0.426 | 0.352 | 0.222 | 0.347 |
| **MACRO** | **0.486** | **0.552** | **0.504** | **0.436** |

Distritos evaluables: 21. Celdas test: 1307.

## Deltas pareados (bootstrap sobre distritos, B=2000)

| comparación | Δ macro ρ [CI 95%] | ¿CI excluye 0? |
|:--|:--|:--|
| transfer − prior pob. | +0.0493 [+0.0057, +0.0963] | SÍ |
| transfer − persistencia | -0.0100 [-0.0741, +0.0565] | no |
| local − transfer | +0.0667 [+0.0290, +0.1064] | SÍ |

Referencias Lima (mismo protocolo): bloque completo M4c 0.464; prior histórico 0.394; per cápita 0.349.

## Lectura

1. **El conocimiento estructural aprendido en Lima transfiere.** El modelo
   Lima→AQP supera significativamente al prior poblacional (+0.049, CI excluye
   0) y empata estadísticamente con la persistencia local — sin haber visto un
   solo dato de Arequipa. Retiene el 88% del desempeño del modelo local
   (0.486/0.552). La relación forma-urbana→concentración-de-crimen no es un
   artefacto limeño.

2. **En la categoría rara, transferir GANA al modelo local** (secuestro:
   transfer 0.426 > local 0.352 > persistencia 0.222). Con ~77 secuestros/año
   en AQP el modelo local no tiene de dónde aprender; el volumen de Lima
   aporta la regularidad que los datos locales no alcanzan. Mismo patrón que
   la fusión attention en Lima: la señal externa ayuda donde la local escasea.

3. **Arequipa es una ciudad "más fácil" que Lima**: hasta el prior poblacional
   (0.436) rinde ahí cerca del techo tabular limeño (0.464). Mejor
   geocodificación (85.5%), ciudad compacta, concentración más marcada. La
   comparación honesta entre ciudades es contra sus propios baselines, no
   entre valores absolutos.

4. **Caveats**: un solo año de test (2023), 21 distritos (CIs anchos), el
   bloque transferible excluye REDATAM/metro/BRT/SV/Sentinel (18 features),
   y ambos oráculos son denuncias geocodificadas (sesgo de denuncia presente
   en ambas puntas — la transferencia del DE-SESGO con r de ENAPRES CIUDADSEG
   queda como extensión natural, no cubierta aquí).
