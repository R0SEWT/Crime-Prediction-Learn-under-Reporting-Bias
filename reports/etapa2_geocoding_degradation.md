# Degradación sintética de geocodificación: curva de evaluabilidad (slr-dj7)

**Diseño:** adelgazamiento binomial de los conteos del oráculo de Lima (base 83.4%) hacia tasas menores, 10 seeds por nivel. El modelo tabular techo (M4c) se RE-ENTRENA sobre los targets degradados; la persistencia usa el historial degradado. 'Medible' = rho intra contra el oráculo degradado (lo que vería el analista); 'real' = contra el oráculo completo (skill verdadero). Supuesto declarado: pérdida aleatoria — cota optimista frente a la pérdida selectiva real (Trujillo).

| tasa geocod. | ρ medible (±sd) | ρ real (±sd) | ρ persistencia | % seeds features>persist |
|--:|:--|:--|--:|--:|
| 83% | 0.464 (±0.000) | 0.464 (±0.000) | 0.401 | 100% |
| 70% | 0.452 (±0.008) | 0.461 (±0.007) | 0.386 | 100% |
| 50% | 0.429 (±0.008) | 0.459 (±0.005) | 0.359 | 100% |
| 25% | 0.384 (±0.013) | 0.450 (±0.010) | 0.302 | 100% |
| 10% | 0.316 (±0.014) | 0.433 (±0.009) | 0.218 | 100% |

## Lectura

1. **El skill real apenas se degrada** (0.464 → 0.450 a nivel Trujillo-25%):
   el modelo estructural entrena bien sobre targets adelgazados — las features
   no dependen del registro. **Lo que colapsa es el skill MEDIBLE**
   (0.464 → 0.384 a 25%; → 0.316 a 10%): a nivel Trujillo, un analista
   subestimaría el desempeño real en ~0.07 de ρ. La penalidad de la mala
   geocodificación cae sobre la *evaluación*, no sobre el modelo.
2. **La persistencia se degrada mucho más rápido** (0.401 → 0.302 a 25%):
   sufre doble (historial de entrenamiento Y referencia de evaluación
   adelgazados). La brecha features−persistencia se ENSANCHA con el deterioro
   del registro (0.063 → 0.082 → 0.098): justo donde el registro es peor, la
   práctica estándar (hotspots por persistencia) pierde más frente al modelo
   estructural. Features > persistencia en el 100 % de seeds a todo nivel.
3. **Matiz al caso Trujillo**: bajo pérdida ALEATORIA, evaluar a 25 % es
   posible pero subestima. El descalificador real de Trujillo es que su
   pérdida es presumiblemente SELECTIVA (por comisaría/zona), lo que sesga la
   evaluación en dirección desconocida — este experimento acota el mejor caso,
   no el real. Se declara como cota optimista.
