# Multiplicidad del ladder: ¿sobrevive el escalón de manzana? (slr-p1u.18, A4)

**Protocolo:** train 2019–2022, test 2023, oráculo geocodificado, HGB con params del ladder. Bootstrap **compartido** sobre distritos dentro de categoría, B=20,000, seed 20260731. Los cuatro escalones se evalúan sobre el MISMO remuestreo en cada réplica, así que las réplicas conservan la correlación entre contrastes. Pares (categoría, distrito) alineados en los cuatro escalones: **194**.

## 1. Contrastes, marginal vs simultáneo

| # | contraste | Δρ | CI 95% marginal | CI 95% **simultáneo** (max-t) | p Westfall-Young |
|:--|:--|--:|:--|:--|--:|
| 1 | H1 manzana censal | +0.0749 | [+0.0476, +0.1019] | [+0.0425, +0.1073] | < 5.0e-05 |
| 2 | H2 forma urbana OSM | +0.0039 | [-0.0188, +0.0270] | [-0.0233, +0.0310] | 0.7346 |
| 3 | H3 ocho fuentes (conjunto) | +0.0091 | [-0.0092, +0.0284] | [-0.0134, +0.0317] | 0.5017 |
| 4 | H4 transporte sobre OSM *(fuera de la familia primaria)* | +0.0053 | [-0.0115, +0.0223] | — | 0.7572 |

Valor crítico simultáneo max-t al 95%: **c = 2.348** (un CI marginal usaría ≈1.96; el precio de la simultaneidad sobre 3 contrastes correlacionados es ese 1.20×).

## 2. Ajustes por multiplicidad (familia primaria de 3)

| contraste | p marginal | p Holm-Bonferroni | p Westfall-Young | ¿sobrevive α=0.05? |
|:--|--:|--:|--:|:--|
| H1 manzana censal | < 5.0e-05 | 0.0001 | < 5.0e-05 | **SÍ** |
| H2 forma urbana OSM | 0.7346 | 0.7346 | 0.7346 | no |
| H3 ocho fuentes (conjunto) | 0.3419 | 0.6839 | 0.5017 | no |

## 3. Tolerancia de familia del escalón de manzana

En las 20,000 réplicas NINGUNA alcanzó el |t| observado del escalón de manzana, así que su p marginal está en el piso de resolución del bootstrap, p ≤ 5.00e-05 = 1/(B+1); todo lo que sigue usa ese piso y por lo tanto **subestima** la tolerancia real.

Bajo Bonferroni el escalón sigue excluyendo 0 para cualquier familia de hasta **m ≈ 1,000 contrastes simultáneos**. La familia primaria tiene 3; incluso la lectura más hostil —contar cada fuente del panel, cada celda del factorial, cada ciudad y cada nivel de thinning como un contraste más— no se acerca a ese orden de magnitud.

## 4. Lectura

El veredicto del ladder **no era un artefacto de multiplicidad**. Con control de error familiar por remuestreo, el escalón de manzana censal sigue siendo significativo y los otros dos siguen sin serlo: la conclusión 'exactamente un escalón real' se mantiene *después* de pagar el precio de las tres pruebas, no solo antes.

Vale registrar la asimetría lógica: el control de multiplicidad **fortalece** el lado positivo del claim (el escalón sobrevive) pero **no** convierte los dos no-significativos en evidencia de ausencia. Que OSM y las ocho fuentes restantes no crucen el umbral —ni marginal ni simultáneo— sigue siendo compatible con efectos pequeños que este diseño no puede resolver; el CI simultáneo, más ancho que el marginal, lo dice explícitamente.
