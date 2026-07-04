# Anti-falsación del claim L3=0 (bead slr-kuk)

_Levanta AUDIT-002: blindar el claim central del SLR —ningún trabajo integra la corrección de cifra negra DENTRO de un modelo predictivo DL (L3)— contra la objeción "no lo hallaste por tu query / criterio journal-only". Búsqueda S2 **sin filtro de venue** (incluye conferencias y preprints). Regenera: `python3 scripts/antifalsify_l3.py`. CSV: `data/datasets/silver/slr/antifalsify_l3_candidates.csv`._

## 1. Diseño de la búsqueda

Se lanzaron **4 consultas booleanas independientes** sobre `/paper/search/bulk` (todos los tipos de venue), cubriendo la conjunción **DL × crimen × corrección-de-reporte** con sinónimos (under-report\*, dark figure, cifra negra, reporting bias/rate, MNAR, victimization survey, propensity to report, latent/true crime, target/label correction). El criterio es de **alto recall**: se prefiere recuperar de más y filtrar, no de menos.

- Totales S2 reportados por query: `[281, 6964]`.
- Papers únicos cribados: **1779**.
- De ellos, **1165 (65%) son no-journal** (conferencia/preprint) — exactamente el material que el SLR journal-only excluye y que un reviewer DL exigiría revisar.
- Ya presentes en el corpus de 40: 11.

## 2. Auto-coding heurístico (near-miss a L3)

Sobre título+abstract se marca un *near-miss L3* cuando coexisten (a) señal de método **DL** y (b) señal de **crimen** y (c) señal de consciencia del **reporte/sub-reporte**. Deliberadamente **no** se exige que dispare además la regex de *integración* (target/label correction, propensity-weighted loss, MNAR-aware, survey-calibrated, joint modelling): ese vocabulario L3 es raro y casi nunca aparece en abstracts, así que usarlo como filtro **pre-descartaría** justo los candidatos que deben ir a lectura humana → haría el "0 L3" vacuo. La integración se usa solo como señal de orden. Es un **filtro de recall** (alto), no una confirmación: L3 se confirma leyendo el paper.

- **Near-miss L3 fuera del corpus: 12.**
- L2-ish (caracterizan el reporte sin integración DL aparente): 184.

## 3. Tabla de evidencia negativa (shortlist a adjudicar)

| venue | año | título | cit |
|---|---|---|---|
| journal | 2017 | State Intimate Partner ViolenceRelated Firearm Laws and Intimate Partner Homicide Rates in | 117 |
| journal | 2015 | Through the Looking Glass | 39 |
| conf/preprint | 2013 | Ecologia social do medo: avaliando a associação entre contexto de bairro e medo de crime | 16 |
| conf/preprint | 2022 | Decisions, Decisions: An Analysis of Identity Theft Victims’ Reporting to Police, Financia | 14 |
| journal | 2019 | Reality versus rhetoric: Assessing the efficacy of third-party hate crime reporting centre | 11 |
| journal | 2021 | Using National Data to Inform Our Understanding of Family and Intimate Partner Violence Vi | 7 |
| journal | 2021 | Preventing outlaw biker crime in the Netherlands or just changing the dark figure? Estimat | 4 |
| journal | 2025 | Likelihood-Free Estimation for Spatiotemporal Hawkes processes with missing data and appli | 2 |
| conf/preprint | 2023 | Geosemantic Surveillance and Profiling of Abduction Locations and Risk Hotspots Using Prin | 0 |
| journal | 2025 | True Crime Podcasting as Journalistic Heterodoxy: Boundary Practices and Journalistic Epis | 0 |
| journal | 2025 | The Intersectionality of Asian Americans’ Violent and Non-Violent Victimization Before and | 0 |
| conf/preprint | 2018 | The CSI Imaginary | 0 |

Cada fila requiere **lectura manual** para confirmar/descartar L3. Si alguno integra genuinamente el mecanismo de reporte en el modelo → revisar el claim del manuscrito (regla L3 del esquema).

## 4. Veredicto y uso en el manuscrito

El claim L3=0 **requiere adjudicar los near-miss antes de afirmarse**. Independientemente del recuento exacto, este ejercicio permite **reformular el claim con precisión defendible**: *"en una búsqueda amplia que incluye conferencias y preprints (N=1779, 65% no-journal), no se identifica ningún trabajo que integre la corrección de sub-reporte en un modelo predictivo DL"* — lo que neutraliza tanto la objeción de query como la de journal-only (AUDIT-002 / AUDIT-003).

**Límite honesto:** el auto-coding es por título+abstract; un L3 que no use el vocabulario esperado podría escaparse. Por eso el shortlist va a lectura humana y la conclusión se enuncia como *no se identificó*, no como *no existe*.

## 5. Adjudicación manual del shortlist (lectura de abstracts)

Leídos los 12 *near-miss*. **Un solo caso** integra genuinamente el mecanismo de
sub-reporte en un modelo predictivo; el resto son falsos positivos de co-ocurrencia
de keywords.

### El único near-miss real — y por qué *afina* el claim en vez de romperlo
**"Likelihood-Free Estimation for Spatiotemporal Hawkes Processes with Missing Data"
(2025, arXiv).** El abstract es explícito: la *missingness por no-reporte* sesga los
parámetros del modelo predictivo y degrada los pronósticos de hotspots, y proponen
estimación likelihood-free para corregirlo. Eso **sí** es integrar el reporte en un
modelo predictivo. Pero:
1. Es un **proceso de Hawkes** (point-process auto-excitante), **no deep learning** →
   no falsa el claim *DL-específico* del SLR.
2. Es un **preprint de arXiv** → fuera del criterio journal-only (IC1) por diseño.

> Cita: Das, P., Banerjee, M., & Sun, Y. (2025). *Likelihood-Free Estimation for
> Spatiotemporal Hawkes Processes with Missing Data and Application to Predictive
> Policing.* arXiv:2502.07111. DOI 10.48550/arXiv.2502.07111.

**Implicación (precisión a la AUDIT-001/002):** el gap L3 se sostiene para **deep
learning** incluso ampliando a conferencias y preprints; pero la idea más amplia
(corregir no-reporte dentro de un modelo predictivo espacio-temporal) **está
emergiendo** (preprint 2025, no-DL). → (a) **escopar el claim a "modelos DL"**, no a
"cualquier modelo predictivo"; (b) **citar este Hawkes paper** como trabajo adyacente
más cercano y evidencia de oportunidad/timeliness de la tesis (preempta la objeción
"se te escapó").

### Falsos positivos (no L3) — motivo de descarte
- *Geosemantic … Abduction Hotspots* (2023, conf.): usa NLP+geoparsing para **generar
  datos** sorteando el sub-reporte (newspaper geoparsing), no integra una tasa de
  reporte en el modelo → no L3.
- Resto (firearm-law IPH epidemiology, sexual-assault reporting survey, fear-of-crime
  multilevel regression, identity-theft victim NCVS study, hate-crime third-party
  reporting policy, IPV data-sources review, outlaw-biker prosecution effects,
  *true-crime podcasting*, Asian-American victimization neg-binomial, *The CSI
  Imaginary* cultural history): estudios de criminología/victimología o cultura sin
  modelo DL; disparan la heurística por co-ocurrencia espuria de "neural"/"deep"/
  "spatiotemporal" + "report".

### Veredicto final
**L3 (integración en modelo DL) = 0, robusto a venue** (1779 papers, 65% no-journal).
El claim se mantiene **escopado a deep learning** y se cita el Hawkes-2025 como el
adyacente más cercano. Esto neutraliza AUDIT-002 (objeción de query) y AUDIT-003
(objeción journal-only) a la vez, y de paso cumple la precisión que pedían ambos
reviewers (no sobre-generalizar de "DL" a "todo modelo").
