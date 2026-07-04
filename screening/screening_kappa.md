# Acuerdo inter-evaluador del screening dual (bead slr-b9b.4)

_Levanta AUDIT-001: convierte el screening single-reviewer del manuscrito en **dual** con una métrica de fiabilidad reportable. Une el screening independiente de Christian con la decisión primaria original y calcula Cohen's kappa título/abstract. Regenera: `python3 scripts/screening_kappa.py`. JSON: `data/silver/slr/screening_kappa.json`._

## 1. Mapeo binario (documentado)

El kappa principal es **binario** (pasa a full-text / no pasa); las razones IC/EC se reportan aparte. Mapeo acordado:

| decisión | clase |
|---|---|
| `include_title_abstract`, `uncertain_fulltext` (orig. `accepted`) | **pasa** |
| `exclude`, `duplicate` (orig. `excluded`) | **no pasa** |

## 2. Resultado

- **Cohen's kappa = 0.845** → acuerdo *casi perfecto* (escala Landis & Koch).
- Acuerdo porcentual: **97.7%** (43/44).
- Esperado por azar: 0.853.
- n = 44 registros (universo local recuperable; ver `analysis/screening_universe.md` para el límite de cobertura vs ~280).

> **Caveat de prevalencia (paradoja de kappa):** los marginales están muy desbalanceados (40/44 pasan), así que el azar esperado es alto (pe=0.853) y kappa *subestima* el acuerdo real respecto al 97.7% bruto. Por eso se reportan **ambos**: kappa para comparabilidad y el acuerdo porcentual para magnitud.

Matriz de confusión (filas = original, columnas = Christian):

| | Christian: pasa | Christian: no pasa |
|---|---|---|
| **original: pasa** | 40 | 0 |
| **original: no pasa** | 1 | 3 |

## 3. Desacuerdos (insumo de adjudicación, slr-b9b.5)

| record | original | Christian | conf. | título | nota |
|---|---|---|---|---|---|
| SLR1-0044 | excluded | uncertain_fulltext | low | Estimating Crime Counts and Characteristics from N | Titulo sobre conteos/caracteristicas NIBRS; sin abstract local no se puede confirmar ML/DL o reporting-bias desde titulo |

Estos casos pasan a **adjudicación** (slr-b9b.5): resolver por discusión y registrar la decisión final.

## 4. Razones IC/EC (descriptivo) y confianza

- Razones primarias de Christian: `{'IC4': 43, 'EC6': 1}`.
- Distribución de confianza: `{'high': 29, 'medium': 14, 'low': 1}`.

## 5. Límite honesto

El kappa se computa sobre el **universo local recuperable** (n=44), no sobre los ~280 registros originales de Ronda 1 (cuyos exports crudos no se conservaron — la causa raíz de AUDIT-001). Por eso mide la fiabilidad de la decisión *include/exclude* sobre los registros que entraron al tracker durable, y debe describirse así en el manuscrito. Recuperar los ~280 sigue pendiente (límite declarado, no inflado).
