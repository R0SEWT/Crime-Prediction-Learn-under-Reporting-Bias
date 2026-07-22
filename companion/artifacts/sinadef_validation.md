# Validación convergente con SINADEF homicidios

> ⚠️ **CORREGIDO 2026-07-14 (slr-x50g).** Este reporte antes correlacionaba una **tasa**
> (homicidios/100k) contra un **conteo** (denuncias absolutas), y publicaba
> ρ(observado)=+0.336 / ρ(latente)=+0.292. Eso medía **población**:
> ρ(población, denuncias)=+0.914 × ρ(población, tasa de homicidios)=+0.326 ≈ 0.30.
> Ahora ambos lados van en tasas por 100k. Los números publicados se conservan en las tablas,
> en la columna «(conteos, publicado)», para que el confundidor quede a la vista.

**Pregunta central.** No: en el pooled total el observado correlaciona más que el latente.

En el total pooled 2019-2024, **tasa contra tasa**:
ρ(observado, homicidios)=`-0.031` y
ρ(latente, homicidios)=`-0.173`; Δ=`-0.142`.
(Como se publicó, tasa-vs-conteo: `0.336` y
`0.297`.)

**El ancla externa NO sostiene ninguna de las dos superficies.** En la comparación correcta (tasa contra tasa) ambas ρ son nulas o negativas, así que esta prueba **no puede** usarse para mitigar la circularidad del resto del diseño.

## Datos usados

- SINADEF silver: `data/silver/sinadef/homicidios_distrito_anio.csv`
- Filtro principal SINADEF: `MUERTE VIOLENTA == HOMICIDIO`.
- Sensibilidad CIE-X agresión X85-Y09: `892` filas vs `8,258` por muerte violenta en el archivo completo; diferencia `-7,366`. En el subconjunto Perú+mapeado: `883` vs `8,104`; diferencia `-7,221`.
- Geografía SINADEF: `COD# UBIGEO DOMICILIO`. SINADEF entregado no contiene UBIGEO de ocurrencia; se usa domicilio del fallecido (residencia != ocurrencia).
- Superficie latente: `43` distritos presentes, años 2019-2024, categorías `estafa, extorsion, robo_hurto_callejero, secuestro, violencia_familiar_sexual`.
- Prefijos UBIGEO en la superficie validada: `1501`.

## Cobertura SINADEF en distritos validados

| año | homicidios | tasa media 100k |
| --- | --- | --- |
| 2019 | 359 | 3.592 |
| 2020 | 302 | 3.045 |
| 2021 | 370 | 3.452 |
| 2022 | 490 | 3.859 |
| 2023 | 588 | 4.342 |
| 2024 | 181 | 1.386 |

## Resultado total

| año/scope | n | ρ obs (tasa) | ρ lat (tasa) | Δ lat-obs | ρ obs (conteos, publicado) | ρ(pob, denuncias) |
| --- | --- | --- | --- | --- | --- | --- |
| 2019 | 43 | 0.015 | -0.030 | -0.045 | 0.287 | 0.918 |
| 2020 | 43 | 0.117 | -0.018 | -0.135 | 0.254 | 0.942 |
| 2021 | 43 | -0.094 | -0.290 | -0.197 | 0.270 | 0.937 |
| 2022 | 43 | -0.022 | -0.232 | -0.211 | 0.388 | 0.942 |
| 2023 | 43 | -0.014 | -0.221 | -0.207 | 0.541 | 0.932 |
| 2024 | 43 | -0.106 | -0.251 | -0.145 | 0.488 | 0.913 |
| pooled | 258 | -0.031 | -0.173 | -0.142 | 0.336 | 0.914 |

Resumen total: el latente mejora al observado en `0/7` scopes con resultado definido.

## Resultado por categoría, pooled

| categoría | n | ρ obs (tasa) | ρ lat (tasa) | Δ lat-obs | ρ obs (conteos, publicado) |
| --- | --- | --- | --- | --- | --- |
| estafa | 258 | -0.313 | -0.292 | 0.021 | 0.124 |
| extorsion | 258 | -0.008 | -0.009 | -0.001 | 0.238 |
| robo_hurto_callejero | 258 | -0.025 | 0.026 | 0.051 | 0.308 |
| secuestro | 258 | 0.064 | 0.163 | 0.100 | 0.289 |
| violencia_familiar_sexual | 258 | 0.208 | 0.163 | -0.045 | 0.398 |

## Resultado por categoría y año

| año | categoría | n | ρ obs (tasa) | ρ lat (tasa) | Δ lat-obs | ρ obs (conteos, publicado) |
| --- | --- | --- | --- | --- | --- | --- |
| 2019 | estafa | 43 | -0.135 | -0.092 | 0.043 | 0.146 |
| 2019 | extorsion | 43 | -0.101 | -0.093 | 0.008 | 0.248 |
| 2019 | robo_hurto_callejero | 43 | -0.063 | -0.015 | 0.049 | 0.275 |
| 2019 | secuestro | 43 | -0.026 | -0.001 | 0.025 | 0.168 |
| 2019 | violencia_familiar_sexual | 43 | 0.241 | 0.131 | -0.110 | 0.349 |
| 2020 | estafa | 43 | -0.305 | -0.334 | -0.029 | 0.050 |
| 2020 | extorsion | 43 | 0.058 | -0.021 | -0.079 | 0.220 |
| 2020 | robo_hurto_callejero | 43 | 0.046 | 0.090 | 0.044 | 0.235 |
| 2020 | secuestro | 43 | 0.053 | 0.093 | 0.039 | 0.241 |
| 2020 | violencia_familiar_sexual | 43 | 0.382 | 0.328 | -0.054 | 0.283 |
| 2021 | estafa | 43 | -0.409 | -0.380 | 0.029 | 0.018 |
| 2021 | extorsion | 43 | 0.081 | 0.120 | 0.039 | 0.287 |
| 2021 | robo_hurto_callejero | 43 | -0.209 | -0.137 | 0.072 | 0.245 |
| 2021 | secuestro | 43 | -0.089 | -0.015 | 0.074 | 0.110 |
| 2021 | violencia_familiar_sexual | 43 | 0.351 | 0.328 | -0.023 | 0.315 |
| 2022 | estafa | 43 | -0.411 | -0.358 | 0.053 | 0.242 |
| 2022 | extorsion | 43 | 0.054 | 0.115 | 0.061 | 0.391 |
| 2022 | robo_hurto_callejero | 43 | 0.048 | 0.099 | 0.051 | 0.376 |
| 2022 | secuestro | 43 | 0.135 | 0.267 | 0.132 | 0.288 |
| 2022 | violencia_familiar_sexual | 43 | 0.086 | 0.035 | -0.051 | 0.408 |
| 2023 | estafa | 43 | -0.348 | -0.321 | 0.027 | 0.325 |
| 2023 | extorsion | 43 | -0.121 | 0.028 | 0.148 | 0.547 |
| 2023 | robo_hurto_callejero | 43 | 0.089 | 0.180 | 0.090 | 0.509 |
| 2023 | secuestro | 43 | 0.126 | 0.368 | 0.241 | 0.548 |
| 2023 | violencia_familiar_sexual | 43 | 0.183 | 0.128 | -0.055 | 0.612 |
| 2024 | estafa | 43 | -0.360 | -0.360 | 0.000 | 0.359 |
| 2024 | extorsion | 43 | -0.018 | -0.117 | -0.098 | 0.548 |
| 2024 | robo_hurto_callejero | 43 | 0.031 | 0.102 | 0.071 | 0.426 |
| 2024 | secuestro | 43 | -0.048 | 0.128 | 0.176 | 0.409 |
| 2024 | violencia_familiar_sexual | 43 | 0.134 | 0.134 | -0.001 | 0.585 |

Resumen por categoría: el latente mejora al observado en `24/35` celdas categoría-scope con resultado definido.

## Caveats

- La validación no es circular respecto de ENAPRES: SINADEF no depende de la decisión de denunciar ante la PNP. Ese es su valor como ancla externa.
- La geografía disponible es domicilio del fallecido, no ocurrencia del homicidio. Si hay desplazamiento residencia-ocurrencia, la correlación distrital se atenúa o se desplaza.
- Homicidio no forma parte de las categorías de-sesgadas (`robo_hurto_callejero`, `estafa`, `extorsion`, `secuestro`, `violencia_familiar_sexual`). La prueba evalúa validez convergente de la superficie total de riesgo, no equivalencia categoría-a-categoría.
- Aunque el enunciado del trabajo habla de Lima/Callao, el archivo latente validado aquí contiene 43 distritos con prefijo `1501`; Callao no entra porque no está en esa superficie.
- El pooled combina distrito-año y debe leerse como resumen descriptivo; no corrige dependencia serial por distrito.
