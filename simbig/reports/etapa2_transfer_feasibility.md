# Factibilidad del test de transferencia espacial (slr-211, prerequisito)

**Fecha:** 2026-07-03 · **Fuente:** corpus nacional MININTER ArcGIS (9,400 chunks,
~9.4M denuncias), filtro del paper (solo_denuncia=1, sin Faltas/NNA), años
2019-2023. Métrica: % de denuncias con `estado_coord = CON COORDENADA`.

## % de geocodificación real por provincia (top volumen)

| provincia | denuncias 2019-2023 | % coord real | ¿oráculo H3 viable? |
|:--|--:|--:|:--|
| LIMA (ref) | 1,406,908 | 83.4 | sí (es el actual) |
| **AREQUIPA** | 221,534 | **85.5** | **SÍ — mejor candidata** |
| CHICLAYO | 187,525 | 69.7 | límite |
| CALLAO | 155,132 | 86.3 | (ya está en el grid) |
| **TRUJILLO** | 154,249 | **24.9** | **NO** |
| PIURA | 137,238 | 83.5 | sí |
| HUANCAYO | 104,434 | 89.7 | sí |
| CUSCO | 83,997 | 80.6 | sí |
| TACNA | 56,991 | 86.4 | sí |

(Tabla completa regenerable escaneando el corpus con pyarrow; ver bead slr-211.)

## Hallazgos

1. **Corrección a la cifra histórica**: el "~31% con coord real" (medido sobre
   LIMA.parquet 1923-2025 completo) NO aplica a los años del panel: en
   2019-2023 Lima está en 83% — la geocodificación mejoró drásticamente.
   Consistente con la mediana de `real_coord_ratio` distrital de 0.798 usada
   en `etapa2_metric_robustness.md`.

2. **El test de transferencia ES viable — pero no donde más duele.**
   Arequipa (85.5%, 221k) permite construir un oráculo H3 tan bueno como el
   de Lima. Piura, Huancayo, Cusco y Tacna también. ENAPRES CIUDADSEG cubre
   estas ciudades → las DOS etapas transfieren (re-estimar r + modelo fino).

3. **Trujillo queda fuera del oráculo fino (24.9%)** — y eso es un hallazgo en
   sí: el epicentro nacional de extorsión es también donde el registro policial
   georreferencia peor. La transferencia a Trujillo solo puede evaluarse a
   nivel distrital (la resolución de Etapa 1), no H3. El caso más interesante
   para el de-sesgo es inalcanzable para el modelo fino — otra cara del mismo
   sesgo de medición que motiva la tesis.

## Qué falta para el test completo (si se ejecuta)

Features mínimas para Arequipa: OSM (Geofabrik ya cubre Perú completo), INEI
manzana (servicio ArcGIS nacional, mismo layer), población INEI (nacional ya
en silver). El bloque M1b+M2 (lo único significativo según
`etapa2_metric_robustness.md`) es reconstruible sin Street View ni Sentinel.
Estimado: 1-2 días de cómputo/integración.
