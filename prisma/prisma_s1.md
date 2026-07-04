# Suplemento S1 — registro PRISMA verificable del corpus (bead slr-8gl)

_Parte **exacta y verificable** de la trazabilidad PRISMA (AUDIT-001): los 40 incluidos con DOI/metadata y los 4 excluidos a full-text con razón. Generado de los stubs `knowledge/papers/*.md` (autoritativos). Los conteos intermedios del flujo (identificación/dedup/título-abstract) son *reconstruidos* (single-reviewer no retuvo cada excluido); el re-screening dual con logging (slr-b9b) se ejecutó sobre el universo recuperable y se reporta en §5. Regenera: `python3 scripts/build_prisma_s1.py`._

## 1. Verificación de tallies (manuscrito vs stubs)

Cruce automático de los conteos que el manuscrito declara como exactos (doc/slr-paper.typ:203) contra los stubs reales:

| cantidad | stubs | manuscrito | |
|---|---|---|---|
| incluidos | 40 | 40 | ✓ |
| excluidos full-text | 4 | 4 | ✓ |
| quartil Q1 | 33 | 33 | ✓ |
| quartil Q2 | 7 | 7 | ✓ |
| año 2023 | 7 | 7 | ✓ |
| año 2024 | 9 | 9 | ✓ |
| año 2025 | 15 | 15 | ✓ |
| año 2026 | 9 | 9 | ✓ |

**Todos los tallies declarados cuadran** con los stubs → los números exactos del manuscrito (n=40, Q1=33/Q2=7, distribución por año, 4 exclusiones) son **reproducibles desde este suplemento**.

- DOIs faltantes: ninguno (todos los registros son verificables).
- Incluidos fuera de Q1/Q2: ninguno (cumple IC1).

## 2. Estudios incluidos (n=40)

| # | autor (año) | título | revista | Q | bases | RQs | L | DOI |
|---|---|---|---|---|---|---|---|---|
| 1 | Deng et al. (2023) | Crime risk prediction incorporating geographical spatiotempo | Information Sciences | Q1 | Scopus | 1 | L0 | 10.1016/j.ins.2023.119414 |
| 2 | Hakyemez et al. (2023) | Incorporating park events into crime hotspot prediction on s | Applied Soft Computing | Q1 | Scopus | 1,3 | L0 | 10.1016/j.asoc.2023.110886 |
| 3 | Jin et al. (2023) | Urban hotspot forecasting via automated spatio-temporal info | Applied Soft Computing | Q1 | Scopus | 1 | L0 | 10.1016/j.asoc.2023.110087 |
| 4 | Mithoo et al. (2023) | Social network analysis for crime rate detection using Spize | Knowledge-Based Systems | Q1 | CrossRef | 1,3 | L0 | 10.1016/j.knosys.2023.110450 |
| 5 | Tam et al. (2023) | Multimodal Deep Learning Crime Prediction Using Tweets | IEEE Access | Q2 | CrossRef | 1,3 | L0 | 10.1109/access.2023.3308967 |
| 6 | Zhou et al. (2023) | Unsupervised Domain Adaptation for Crime Risk Prediction Acr | IEEE Transactions on Computational Social Systems | Q1 | CrossRef | 1 | L0 | 10.1109/tcss.2022.3207987 |
| 7 | Wu et al. (2023) | Auditing the fairness of place-based crime prediction models | Computers, Environment and Urban Systems | Q1 | ScienceDirect | 1 | L0 | 10.1016/j.compenvurbsys.2023.101967 |
| 8 | Yang et al. (2024) | A Reinforcement Learning Approach Combined With Scope Loss F | IEEE Access | Q2 | CrossRef | 1,3 | L0 | 10.1109/access.2024.3473296 |
| 9 | Brunton-Smith et al. (2024) | Estimating the Reliability of Crime Data in Geographic Areas | The British Journal of Criminology | Q1 | Scopus | 2 | L2 | 10.1093/bjc/azae018 |
| 10 | Fé et al. (2024) | Partial Identification of the Dark Figure of Crime with Surv | Journal of Quantitative Criminology | Q1 | Scopus | 2 | L2 | 10.1007/s10940-024-09593-4 |
| 11 | Gu et al. (2024) | Graph Representation Learning for Street-Level Crime Predict | ISPRS International Journal of Geo-Information | Q2 | Scopus | 1,3 | L0 | 10.3390/ijgi13070229 |
| 12 | Cesario et al. (2024) | Multi-density crime predictor: an approach to forecast crimi | Journal of Big Data | Q1 | CrossRef | 1 | L0 | 10.1186/s40537-024-00935-4 |
| 13 | Jing et al. (2024) | A deep multi-scale neural networks for crime hotspot mapping | Computers, Environment and Urban Systems | Q1 | Scopus | 1 | L0 | 10.1016/j.compenvurbsys.2024.102089 |
| 14 | Lee et al. (2024) | Analysing non-linearities and threshold effects between stre | Urban Studies | Q1 | Scopus | 3 | L0 | 10.1177/00420980241270948 |
| 15 | Timukaite et al. (2024) | Trust in the police and crime reporting: Reassessing assumpt | International Review of Victimology | Q1 | Scopus | 2 | L2 | 10.1177/02697580241296569 |
| 16 | Zhou et al. (2024) | HDM-GNN: A Heterogeneous Dynamic Multi-view Graph Neural Net | ACM Transactions on Sensor Networks | Q1 | ACM DL, Scopus | 1,3 | L0 | 10.1145/3665141 |
| 17 | Albors Zumel et al. (2025) | Deep Learning for Crime Forecasting: The Role of Mobility at | Journal of Quantitative Criminology | Q1 | Scopus | 1,3 | L0 | 10.1007/s10940-025-09629-3 |
| 18 | Boldt et al. (2025) | GraphTrace: A Graph-Guided Hotspot Detection Method for CCTV | Journal of Quantitative Criminology | Q1 | CrossRef | 1 | L0 | 10.1007/s10940-025-09623-9 |
| 19 | Wheeler et al. (2025) | Using Victimization Reporting Rates to Estimate the Dark Fig | Crime and Delinquency | Q2 | CrossRef | 2 | L2 | 10.1177/00111287251359210 |
| 20 | Devi et al. (2025) | A rotation invariant Co-ordinate convolutional neural networ | Expert Systems with Applications | Q1 | CrossRef | 1 | L0 | 10.1016/j.eswa.2025.128539 |
| 21 | Fan et al. (2025) | Research on a Crime Spatiotemporal Prediction Method Integra | Big Data and Cognitive Computing | Q1 | Scopus | 1,3 | L0 | 10.3390/bdcc9070179 |
| 22 | Fu et al. (2025) | Augmented graph information bottleneck with type-aware perio | Information Processing & Management | Q1 | Scopus | 1 | L0 | 10.1016/j.ipm.2025.104227 |
| 23 | Jin et al. (2025) | An Event-Centric Framework for Predicting Crime Hotspots Wit | IEEE Transactions on Knowledge and Data Engineering | Q1 | CrossRef | 1,3 | L0 | 10.1109/tkde.2025.3618389 |
| 24 | Kim et al. (2025) | Crime mapping in urban environments using explainable AI: A  | Sustainable Cities and Society | Q1 | ScienceDirect | 1,3 | L0 | 10.1016/j.scs.2025.106507 |
| 25 | Liang et al. (2025) | Spatiotemporal crime prediction and fairness-constrained spa | International Journal of Applied Earth Observation and Geoinformation | Q1 | Scopus | 1 | L0 | 10.1016/j.jag.2025.104973 |
| 26 | İlgün et al. (2025) | Exploratory data analysis, time series analysis, crime type  | Neural Computing and Applications | Q1 | CrossRef | 1 | L0 | 10.1007/s00521-025-11094-9 |
| 27 | Shahmoradi et al. (2025) | Hybrid ST-ResNet and LSTM approach for precise crime hotspot | Scientific Reports | Q1 | Scopus | 1,3 | L0 | 10.1038/s41598-025-24559-7 |
| 28 | Shan et al. (2025) | Ada-GCNLSTM: An adaptive urban crime spatiotemporal predicti | Journal of Safety Science and Resilience | Q1 | ScienceDirect | 1 | L0 | 10.1016/j.jnlssr.2024.11.003 |
| 29 | Butt et al. (2025) | START: A Spatiotemporal Autoregressive Transformer for Enhan | IEEE Transactions on Computational Social Systems | Q1 | CrossRef | 1 | L0 | 10.1109/tcss.2025.3550196 |
| 30 | Wang et al. (2025) | MRAGNN: Refining urban spatio-temporal prediction of crime o | Expert Systems with Applications | Q1 | ScienceDirect | 1,3 | L0 | 10.1016/j.eswa.2024.125940 |
| 31 | Yue et al. (2025) | Interpretable spatial machine learning for understanding spa | Applied Geography | Q1 | CrossRef | 1,3 | L0 | 10.1016/j.apgeog.2024.103503 |
| 32 | Shiraptini et al. (2026) | AI and Machine Learning-Enabled Cognitive Digital Twin for C | IEEE Access | Q2 | CrossRef | 1 | L0 | 10.1109/access.2026.3658944 |
| 33 | Bhumika et al. (2026) | FedCrime: Zero-inflation adaptive federated learning for cri | Neurocomputing | Q1 | Scopus | 1 | L1 | 10.1016/j.neucom.2026.133217 |
| 34 | Guo et al. (2026) | Multimodal spatio-temporal fusion: A generalizable GCN-LSTM  | Information Fusion | Q1 | ScienceDirect | 1,3 | L0 | 10.1016/j.inffus.2026.104164 |
| 35 | Guo et al. (2026) | How well do street view images predict crime rates in London | Computers, Environment and Urban Systems | Q1 | Scopus | 3 | L0 | 10.1016/j.compenvurbsys.2025.102390 |
| 36 | Hakyemez et al. (2026) | Enhancing Deep Learning-based Crime Hotspot Predictions With | Applied Spatial Analysis and Policy | Q2 | Scopus | 1,3 | L0 | 10.1007/s12061-025-09789-6 |
| 37 | Moradi et al. (2026) | Theory-guided agent-based modeling for crime prediction and  | Expert Systems with Applications | Q1 | CrossRef | 1 | L0 | 10.1016/j.eswa.2025.129181 |
| 38 | Moreira et al. (2026) | Unequal crime reporting: trust in the police and racial disp | Policing: An International Journal | Q2 | Scopus | 2 | L2 | 10.1108/PIJPSM-02-2026-0056 |
| 39 | Palma-Borda et al. (2026) | Cooperative patrol routing: Optimizing urban crime surveilla | Engineering Applications of Artificial Intelligence | Q1 | CrossRef | 1 | L0 | 10.1016/j.engappai.2025.113706 |
| 40 | Vairetti et al. (2026) | Deep learning for crime analytics: A prioritization system f | Expert Systems with Applications | Q1 | CrossRef | 1,3 | L1 | 10.1016/j.eswa.2026.131694 |

## 3. Excluidos a full-text con razón documentada (n=4)

| autor (año) | título | revista | DOI | razón de exclusión |
|---|---|---|---|---|
| briz_redon2024_bayesian_aoristic (2024) | A Bayesian Aoristic Logistic Regression to Model Spatio | Journal of Quantitative Criminology | 10.1007/s10940-023-09580-1 | IC4 no cumplido: regresión logística bayesiana, no ML/DL. Aborda incertidumbre temporal en el momento del crimen (intervalo-censura), no sesgo de denuncia/dark figure. |
| rezvani2024_smart_hotspot_flood (2024) | Smart Hotspot Detection Using Geospatial Artificial Int | Sustainable Cities and Society | 10.1016/j.scs.2024.105873 | EC6: el objeto de estudio principal es riesgo de inundación (flood risk), no crimen urbano. Encontrado por coincidencia de términos 'hotspot detection' en búsqueda de crimen. |
| liu2025_poi_spatial_crime (2025) | Capturing the spatial arrangement of POIs in crime mode | Computers, Environment and Urban Systems | 10.1016/j.compenvurbsys.2024.102245 | IC4 no cumplido: usa regresión binomial negativa, no ML/DL. Introduce métricas de arreglo espacial de POIs (n_SVDE, ANN_ratio) pero las evalúa con modelos estadísticos de regresión, no con modelos de predicción ML/DL. |
| lohr2026_nibrs_crime_counts (2026) | Estimating Crime Counts and Characteristics from NIBRS  | Journal of Quantitative Criminology | 10.1007/s10940-025-09650-6 | IC4 no cumplido: metodología estadística de muestreo para generar estimaciones nacionales desde la base NIBRS (National Incident-Based Reporting System de EEUU). Sin componente ML/DL ni predicción espacio-temporal de crimen. No aborda explícitamente cifra negra/dark figure como objeto de estudio. |

## 4. Screening dual independiente (slr-b9b)

Para levantar AUDIT-001, el universo **recuperable** (n=44) fue re-cribado de forma **independiente** por un segundo evaluador (C. Velasquez) a nivel título/abstract, con logging por registro. Acuerdo inter-evaluador (binario pasa/no-pasa a full-text):

- **Cohen's κ = 0.845** (casi perfecto, Landis & Koch); acuerdo porcentual **97.7%** (43/44).
- Caveat de prevalencia (paradoja de κ): marginales desbalanceados (pe=0.853) → se reportan κ y acuerdo % juntos.
- Desacuerdos: **1**.

Adjudicación de desacuerdos (decisiones individuales preservadas, no sobrescritas):

| record | original | Christian | final | razón | resuelto por |
|---|---|---|---|---|---|
| SLR1-0044 (lohr2026_nibrs_crime_counts) | excluded | uncertain_fulltext | **excluded** | IC4: el trabajo estima conteos/caracteristicas de crimen desde NIBRS por muestreo y estima… | consenso (Rody Vilchez + Christian Velasquez) |

→ Tras adjudicación, el corpus de incluidos permanece en **40** (la única discrepancia, lohr2026, se resolvió manteniendo la exclusión IC4). El screening dual **confirma** la selección original.

## 5. Alcance y límite honesto

Este S1 cubre la trazabilidad **verificable**: cada incluido y cada excluido-con-razón es auditable por DOI, y el screening incluye-vs-excluye sobre el universo recuperable fue **re-cribado de forma dual** (§4). Lo que **no** se recupera son los conteos intermedios del flujo PRISMA sobre los ~280 registros originales de Ronda 1 (identificación → dedup → exclusión en título/abstract): esos exports crudos no se conservaron uno-por-registro, así que esas cifras siguen *reconstruidas* en @fig-prisma. El re-screening dual mide la fiabilidad de la decisión sobre el universo durable (n=44), no sobre los ~280 → límite declarado, no inflado.
