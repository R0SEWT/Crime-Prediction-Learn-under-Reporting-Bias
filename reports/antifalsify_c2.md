# Anti-falsacion C2: placebo-controlled label intervention

_Bead `slr-p1u.7`. Regenera con `python3 scripts/antifalsify_c2.py --surface all`. El objetivo es intentar tumbar el claim de novedad C2 antes del envio SIMBig, manteniendo artefactos reproducibles y sin MCPs._

## 1. Diseno

Se separan dos superficies. **Surface A (S2)** cubre literatura indexada y no-journal via Semantic Scholar, reusando `scripts/s2_client.py`. **Surface B (Peru gris)** cubre tesis y repositorios peruanos via ALICIA/CONCYTEC; RENATI se trata como metadato cosechado cuando aparece, porque el sitio directo puede estar protegido. La adjudicacion de C2 exige mas que prediccion de delitos: debe haber intervencion de label/target con placebo, permutacion, control negativo o corrupcion/aleatorizacion equivalente.

## 2. Surface A - Semantic Scholar

- Totales S2 reportados por query: `[996, 16156, 5]`.
- Papers unicos cribados: **1005**.
- No-journal/preprint/conference: **164 (16%)**.
- Ya presentes en corpus local: **11**.
- Near-field crime prediction + ML/DL: **409**.
- Near-miss C2 heuristico: **0**.

| year | venue | title | c2 | cit |
|---|---|---|---:|---:|
| 2017 | Neural Information Processing Sy | Counterfactual Fairness | 0 | 1896 |
| 2017 | PLoS ONE | Prediction of crime occurrence from multi-modal data using deep learning | 0 | 230 |
| 2017 |  | Securing with algorithms: Knowledge, decision, sovereignty | 0 | 183 |
| 2018 | International Conference on Info | DeepCrime: Attentive Hierarchical Recurrent Networks for Crime Prediction | 0 | 168 |
| 2017 | Physica A: Statistical Mechanics | Crime prediction through urban metrics and statistical learning | 0 | 168 |
| 2021 | Visual Computing for Industry, B | Crime forecasting: a machine learning and computer vision approach to crime prediction and | 0 | 153 |
| 2018 | IEEE Annual Information Technolo | Crime Analysis Through Machine Learning | 0 | 123 |
| 2020 | Journal of quantitative criminol | Mapping the Risk Terrain for Crime Using Machine Learning | 0 | 112 |
| 2017 | Chinese Annals of Mathematics. S | Deep Learning for Real-Time Crime Forecasting and Its Ternarization | 0 | 94 |
| 2019 | IEEE technology & society magazi | AI Ethics in Predictive Policing: From Models of Threat to an Ethics of Care | 0 | 84 |
| 2021 | IEEE Access | An Empirical Analysis of Machine Learning Algorithms for Crime Prediction Using Stacked Ge | 0 | 84 |
| 2023 | Journal of Ambient Intelligence  | Machine learning in crime prediction | 0 | 80 |
| 2022 | International Joint Conference o | Multi-Graph Fusion Networks for Urban Region Embedding | 0 | 79 |
| 2018 |  | Crime Prediction & Monitoring Framework Based on Spatial Analysis | 0 | 77 |
| 2020 | COMS2 | Crime Prediction Using Spatio-Temporal Data | 0 | 76 |
| 2017 | arXiv.org | Deep Learning for Real Time Crime Forecasting | 0 | 73 |
| 2020 | Ai & Society | Conservative AI and social inequality: conceptualizing alternatives to bias through social | 0 | 68 |
| 2018 |  | Survey of Analysis of Crime Detection Techniques Using Data Mining and Machine Learning | 0 | 66 |
| 2018 | Journal of Money Laundering Cont | Illicit Bitcoin transactions: challenges in getting to the who, what, when and where | 0 | 47 |
| 2020 | 2020 International Conference on | Crime Prediction Using K-Nearest Neighboring Algorithm | 0 | 46 |
| 2020 | International Conference on Meas | Addressing Crime Situation Forecasting Task with Temporal Graph Convolutional Neural Netwo | 0 | 43 |
| 2022 | International journal of compute | Crime Prediction and Analysis Using Machine Learning | 0 | 42 |
| 2020 | Knowledge-Based Systems | CSAN: A neural network benchmark model for crime forecasting in spatio-temporal scale | 0 | 41 |
| 2023 | Frontiers in Big Data | Big data analytics and smart cities: applications, challenges, and opportunities | 0 | 40 |
| 2020 | Lecture Notes in Electrical Engi | Survey on Crime Analysis and Prediction Using Data Mining and Machine Learning Techniques | 0 | 37 |

## 3. Surface B - Peru gris (tesis/repositorios)

- Registros ALICIA unicos cribados: **91**.
- Near-field prediccion de delitos + ML/DL: **3**.
- Candidatos amplios de prior art local (prediccion/mineria/mapa del delito): **52**.
- Registros con vocabulario C2 en texto institucional extraido: **7**.
- Near-miss C2 heuristico: **0**.
- Repositorios/instituciones mas frecuentes: UCV-Institucional=16, PUCP-Institucional=11, PUCP-Tesis=10, ONPE-Institucional=4, UNHEVAL-Institucional=3, ULIMA-Institucional=3, ULASAMERICAS-Institucional=3, USS-Institucional=3.

| year | institution/repo | format | title | local | c2 | text hit |
|---|---|---|---|---:|---:|---:|
| 2024 | Universidad Católica San Pablo | tesis de grado | Detección automática y análisis de puntos con alta frecuencia (hotspots) de crímenes en ce | 1 | 0 | 0 |
| 2024 | Universidad Nacional Federico Vill | tesis de grado | Identificación de las Zonas con Incidencia de Delincuencia Aplicando Sistema de Informació | 1 | 0 | 0 |
| 2025 | Pontificia Universidad Católica de | tesis de maestría | Red neuronal de inteligencia artificial para optimizar las estrategias de investigación de | 1 | 0 | 0 |

## 4. Adjudicacion deep-research independiente

Se pidio una segunda opinion independiente con agentes paralelos sobre literatura
academica, web/GitHub y literatura gris peruana. Ningun agente encontro un falsador
real de C2. Los near-misses que deben citarse o tenerse presentes son:

| caso | por que esta cerca | por que NO falsa C2 |
|---|---|---|
| Mohler et al. 2015, JASA, randomized field trials of predictive policing | usa ensayo aleatorizado/controlado y discute la imposibilidad de un placebo puro en patrullaje | la intervencion es despliegue/patrullaje, no label/target |
| Akpinar, De-Arteaga & Chouldechova 2021, FAccT | modela denuncia diferencial y sesgo en predictive policing; antecedente mas cercano sobre labels observados | no es placebo-controlado ni permuta/interviene labels como control negativo |
| Ensign et al. 2018, FAT*/PMLR | modela feedback loops por datos policiales descubiertos/observados | no hace intervencion placebo de labels |
| Lum & Isaac 2016, Significance | muestra sesgo de datos policiales en prediccion/patrullaje | no contiene experimento placebo-controlado |
| Fogliato et al. 2020, AISTATS | label noise/fairness en criminal justice risk assessment | no es crime forecasting/predictive policing ni C2 |
| Saunders et al., Chicago Strategic Subject List evaluation | incluye placebo/sensitivity checks econometricos | el placebo es de timing/evaluacion, no de label/target |
| Riascos Villegas et al., underreported spatio-temporal crime | modela crimen sub-reportado | no hay placebo/permutacion de labels |

### Peru gris adjudicado

La superficie peruana si contiene prior art cercano de prediccion/mineria/mapas de
delito: USMP 2015 (prediccion de hechos delictivos en La Molina), USAT 2015
(alerta temprana con mineria de datos en Chiclayo), PUCP 2019 (SIDPOL/Mapa del
Delito), UPC 2022 (servicio web para prediccion de robos), ULima 2024 (modelos de
regresion para zonas de riesgo), UCSP 2024 (hotspots con aprendizaje profundo),
y Continental/WEBIST 2024 (Random Forest para robos/hurtos). Ninguno presenta
placebo, permutacion, etiquetas aleatorizadas, target intervention o control
negativo de labels. Las busquedas SEMAS produjeron ruido/no relacionados; no
emerge como superficie relevante para C2.

## 5. Veredicto operacional

No se identifico ningun falsador C2 en las superficies revisadas. El wording recomendado es **escopado**: *To our knowledge, this is the first study in crime prediction to use a placebo/negative-control label intervention, in which target labels are deliberately perturbed while the modeling pipeline is held fixed.* Limites: S2 usa busqueda capped de alto recall; Peru gris cubre ALICIA/CONCYTEC y busquedas web/RENATI/SEMAS puntuales, no toda literatura gris latinoamericana.

## 6. Artefactos

- S2 CSV: `data/datasets/silver/slr/antifalsify_c2_s2_candidates.csv`
- Peru gray CSV: `data/datasets/silver/slr/antifalsify_c2_peru_gray_candidates.csv`
- Criterio de adjudicacion: C2 requiere prediccion de crimen + intervencion explicita de label/target + placebo/permutacion/control negativo o equivalente.
