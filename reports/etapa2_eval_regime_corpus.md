# Régimen de evaluación en el corpus DL-crimen (slr-p1u.9)

**Fuente:** formularios de extracción del SLR (`knowledge/papers/*.md`, 40
aceptados), subconjunto empírico DL-forecasting con datos EEUU/China (14).
"n/r" = no reportado en la extracción. **Verificación full-text (bronze)
completada 2026-07-04 (slr-p1u.8)**: columnas target/split/CI adjudicadas
contra data/bronze/*.txt; el hit espacial de Jing+2024 era falso positivo
("spatial blocks" arquitecturales) — 0/14 splits espaciales se sostiene. Propósito: tabla de
related work que contextualiza nuestro protocolo (oráculo independiente del
target de entrenamiento, splits espaciales con buffer, priors simples como
baseline, CIs pareados multi-seed).

| paper | datos | unidad espacial | cadencia | target de eval | baselines | ¿CI/seeds? |
|:--|:--|:--|:--|:--|:--|:--|
| Shan+2025 Ada-GCNLSTM | NYC/Chicago/China 2014-17 | grillas 3 km | n/r | conteos policiales | familias GNN/LSTM | no |
| Zhou+2024 HDM-GNN | NYC 2016-20 | 263 vecindarios | n/r | conteos NYPD | ARIMA, LSTM, STGCN... | no |
| Fan+2025 Informer-STGCN | Chicago 2015-20 | 22 distritos policiales | n/r | conteos | LSTM, Ridge, RF | no |
| Jin+2023 ASTIF-Net | SF 2003-18 | grid 20×20 | diaria | conteos | 11 modelos neuronales | no |
| Jing+2024 multiescala | Chicago 2016-17 | 500-2000 m | diaria | conteos (asaltos) | n/r | no |
| Shahmoradi+2025 ST-ResNet+LSTM | Chicago | 500-2000 m | diaria | conteos (hurto) | ST-ResNet | no |
| Hakyemez+2023 park events | Chicago | segmento vial | diaria/intradiaria | conteos | GWNet, STGCN, LSTM | no |
| Hakyemez+2026 SSRS | Chicago Central 2015-18 | segmento vial | diaria/turno | conteos | GWNet, STGCN, STNetKDE | no |
| Albors+2025 mobility | 4 ciudades EEUU 2019-23 | 0.2 km² | 12 horas | conteos | crimen-histórico-solo | **sd sobre 4 seeds** |
| Bhumika+2026 FedCrime | LA/Chicago | n/r | n/r | conteos | baselines FL | **±std 3 seeds + p-values** |
| Wang+2025 MRAGNN | Chicago/LA 2018 | n/r | n/r | conteos (multietiqueta) | GNN | no |
| Fu+2025 ExCrime-GIB | n/r | n/r | n/r | conteos | GNN | no |
| Access2024 RL+scope loss | 2 datasets | n/r | n/r | conteos | "baselines" | **CIs y p-values (self-reported)** |
| İlgün+2025 (NCA) | SF/Chicago/Phila | distrito policial | mensual/serie | conteos | ML/DL/estadísticos | no |

## Observaciones para el manuscrito (§Related Work)

1. **Target: 14/14 evalúan contra conteos policiales observados.** Ninguno
   corrige, modela ni cuantifica el sesgo de denuncia en la *evaluación*
   (algunos lo mencionan como limitación). Coherente con la brecha L3=0 del
   SLR: el mecanismo de reporte no entra ni al modelo ni al protocolo.
2. **Splits: cronológicos cuando se reportan (p.ej. 70/30); NINGUNO usa
   validación espacial** (grouped o buffered). Nuestra medición interna del
   costo de ignorarlo: random-CV infla +0.125 sobre buffered spatial CV.
   El framing correcto no es acusar de random-CV — es la ausencia de control
   por estructura espacial y la sub-especificación del split (n/r frecuente).
3. **Baselines: las familias neuronales se comparan entre sí** (LSTM, STGCN,
   ST-ResNet, GWNet…). Persistencia simple y priors estructurales
   (población/forma urbana) casi nunca aparecen — exactamente los baselines
   que en nuestro régimen de datos ganan al DL. Excepción parcial: Albors+2025
   (crimen-histórico-solo) y los estadísticos de İlgün+2025.
4. **Incertidumbre (CORREGIDO por verificación full-text 2026-07-04,
   slr-p1u.8): 11/14 no reportan ninguna; 3/14 reportan sd por seeds o tests
   no pareados** (Albors+2025: sd sobre 4 seeds; Bhumika+2026: ±std 3 seeds
   con p-values; Access2024: CIs/p-values self-reported). **0/14 reporta
   incertidumbre pareada sobre unidades espaciales** — la práctica que este
   paper introduce. La versión anterior de esta observación ("0/14 CIs o
   multi-seed") era stub-derived y queda supersedida. Nuestra experiencia
   sigue siendo el argumento concreto: el rigor multi-seed tumbó nuestro
   propio mejor resultado (+0.079 "sinérgico" → ruido) antes de que lo
   hiciera un revisor.
5. **Régimen de datos:** cadencia diaria/intradiaria y unidades 0.2-9 km²
   sobre ciudades con registro denso y geocodificación ~completa. La
   transferibilidad de esas conclusiones a ciudades con 25-85 % de
   geocodificación y cadencia anual efectiva es la pregunta que este paper
   responde.

Regenerable: greps sobre los stubs del corpus; tabla curada a mano.
