# Reporte de análisis y evaluación del notebook `tp_cdd.ipynb`

**Dataset:** `i40.csv` — Mediciones operativas para mantenimiento predictivo industrial  
**Fecha de ejecución original:** 2026-05-01  
**Última corrección y re-ejecución:** 2026-05-14  
**Notebook fuente:** `tp1/tp_cdd.ipynb`

---

## Índice

1. [Contexto y estructura general](#1-contexto-y-estructura-general)
2. [Fase 1 — Carga y revisión inicial del dataset](#2-fase-1--carga-y-revisión-inicial-del-dataset)
3. [Fase 2 — Análisis Exploratorio de Datos (EDA)](#3-fase-2--análisis-exploratorio-de-datos-eda)
   - [3.1 Medidas de tendencia central y dispersión](#31-medidas-de-tendencia-central-y-dispersión)
   - [3.2 Análisis de distribuciones](#32-análisis-de-distribuciones)
   - [3.3 Proporciones de variables categóricas](#33-proporciones-de-variables-categóricas)
   - [3.4 Relaciones entre variables y conclusiones del EDA](#34-relaciones-entre-variables-y-conclusiones-del-eda)
4. [Fase 3 — Preprocesamiento para modelado](#4-fase-3--preprocesamiento-para-modelado)
   - [4.1 Limpieza inicial e imputación con KMeans](#41-limpieza-inicial-e-imputación-con-kmeans)
   - [4.2 Evaluación de multicolinealidad](#42-evaluación-de-multicolinealidad)
   - [4.3 Tratamiento de valores atípicos](#43-tratamiento-de-valores-atípicos)
   - [4.4 Codificación, balanceo de clases y normalización](#44-codificación-balanceo-de-clases-y-normalización)
5. [Fase 4 — Modelado y evaluación de clasificadores](#5-fase-4--modelado-y-evaluación-de-clasificadores)
   - [5.1 División entrenamiento-prueba](#51-división-entrenamiento-prueba)
   - [5.2 Estrategia de validación cruzada](#52-estrategia-de-validación-cruzada)
   - [5.3 Modelos entrenados y búsqueda de hiperparámetros](#53-modelos-entrenados-y-búsqueda-de-hiperparámetros)
   - [5.4 Resultados sobre el conjunto de prueba](#54-resultados-sobre-el-conjunto-de-prueba)
   - [5.5 Análisis comparativo del rendimiento](#55-análisis-comparativo-del-rendimiento)
6. [Fase 5 — Notas finales y fundamentación de decisiones](#6-fase-5--notas-finales-y-fundamentación-de-decisiones)
   - [6.1 Preservación de ambas variables de temperatura](#61-preservación-de-ambas-variables-de-temperatura)
   - [6.2 Undersampling como estrategia de balanceo](#62-undersampling-como-estrategia-de-balanceo)
   - [6.3 Imputación bivariada con KMeans para pares correlacionados](#63-imputación-bivariada-con-kmeans-para-pares-correlacionados)
   - [6.4 Resumen de la pipeline completa](#64-resumen-de-la-pipeline-completa)
7. [Errores detectados y correcciones aplicadas](#7-errores-detectados-y-correcciones-aplicadas)
   - [7.1 Error en la normalización: `Normalizer()` en lugar de `StandardScaler()`](#71-error-en-la-normalización-normalizer-en-lugar-de-standardscaler)
   - [7.2 Mensaje de umbral inconsistente en la salida](#72-mensaje-de-umbral-inconsistente-en-la-salida)
   - [7.3 Discrepancia menor en las conclusiones del EDA](#73-discrepancia-menor-en-las-conclusiones-del-eda)
   - [7.4 Execution counts inconsistentes en el notebook original](#74-execution-counts-inconsistentes-en-el-notebook-original)
8. [Conclusiones generales y recomendaciones](#8-conclusiones-generales-y-recomendaciones)

---

## 1. Contexto y estructura general

El presente documento constituye un reporte técnico detallado del trabajo de análisis y modelado contenido en el notebook `tp_cdd.ipynb`, desarrollado como parte de un proyecto de Ciencia de Datos aplicada al mantenimiento predictivo industrial. El objetivo central del trabajo es construir y evaluar modelos de clasificación capaces de anticipar fallas en equipamiento a partir de mediciones operativas registradas durante el funcionamiento normal y anómalo de las máquinas.

El notebook se articula en dos grandes partes técnicas, cada una de ellas subdividida en secciones que avanzan de manera incremental desde la exploración inicial de los datos hasta la evaluación comparativa de los modelos entrenados. La **Parte 1** comprende el Análisis Exploratorio de Datos (EDA), donde se examinan en profundidad las características estadísticas, distribucionales y relacionales de todas las variables disponibles. La **Parte 2** abarca el preprocesamiento completo del dataset y el posterior entrenamiento, optimización de hiperparámetros y evaluación de seis modelos de clasificación pertenecientes a distintas familias algorítmicas: modelos lineales, probabilísticos, basados en instancias, árboles de decisión y ensembles.

El stack tecnológico empleado es el estándar del ecosistema científico de Python. Las operaciones de manipulación y análisis de datos se realizan con `pandas` y `numpy`; las visualizaciones se construyen con `matplotlib` y `seaborn`; y todo el pipeline de modelado —desde la división de datos hasta la evaluación con métricas múltiples— se implementa sobre `scikit-learn`. Para el balanceo de clases se incorpora la librería especializada `imbalanced-learn`. La variable objetivo sobre la que se trabaja es `target`, una variable binaria que indica si la máquina se encuentra en estado de falla (`failure`) o en operación normal (`normal`), a partir de un conjunto de siete variables predictoras: cinco numéricas (temperaturas, velocidad, torque y desgaste) y una categórica con tres niveles (tipo de producto).

---

## 2. Fase 1 — Carga y revisión inicial del dataset

La primera fase del trabajo establece las bases de todo el análisis subsiguiente mediante una inspección estructural del dataset. El propósito de esta etapa es obtener una comprensión inmediata de la magnitud y composición de los datos antes de sumergirse en análisis más sofisticados: cuántas observaciones están disponibles, qué tipo de información contiene cada columna, si existen valores ausentes que requieran tratamiento y qué columnas resultan irrelevantes para el modelado predictivo.

Tras la importación de las librerías necesarias y la configuración de un estilo visual consistente mediante `seaborn.set_theme()`, se carga el archivo `i40.csv` en un DataFrame de pandas y se despliegan sus propiedades fundamentales. El dataset contiene **14.521 filas y 9 columnas**, una dimensión considerable para un problema de clasificación binaria en el contexto de mantenimiento industrial. De las nueve columnas, siete son numéricas —incluyendo dos identificadores (`idx` y `parent_device_id`), tres mediciones de proceso (`air_temp [K]`, `process_temp [K]`, `speed [RPM]`) y dos indicadores de rendimiento (`torque [Nm]`, `tool_wear [min]`)— mientras que las dos restantes son categóricas: `product_type`, que clasifica el producto fabricado en tres tipos (L, M, H), y `target`, la variable objetivo con dos clases (`normal`, `failure`).

Un hallazgo relevante en esta primera inspección es la presencia de **40 valores nulos en la columna `air_temp [K]`**, mientras que el resto de las variables se encuentran completas. Esta observación es importante porque la temperatura del aire es una de las variables con mayor poder predictivo potencial, dada su relación física con la temperatura del proceso. La decisión inicial del equipo es acertada: se identifican `idx` y `parent_device_id` como columnas puramente identificadoras, sin valor informativo para la tarea de clasificación, y se las excluye del subconjunto de variables numéricas que serán objeto del análisis estadístico y del modelado posterior. Esta exclusión temprana evita que estos identificadores contaminen los cálculos de correlación o sean utilizados inadvertidamente como features.

---

## 3. Fase 2 — Análisis Exploratorio de Datos (EDA)

La segunda fase del trabajo constituye el corazón del análisis descriptivo. El EDA se despliega en cuatro movimientos analíticos complementarios: primero, una caracterización estadística univariada mediante medidas de posición, dispersión y forma; segundo, una inspección visual de las distribuciones mediante histogramas, estimaciones de densidad y diagramas de caja; tercero, un examen de las proporciones y el balance de las variables categóricas; y cuarto, una exploración de las relaciones bivariadas —tanto lineales como condicionadas por la clase objetivo— que anticipan la estructura que los modelos de clasificación deberán aprender.

### 3.1 Medidas de tendencia central y dispersión

El análisis estadístico univariado persigue un doble objetivo: por un lado, caracterizar la escala y la variabilidad de cada medición para fundamentar las decisiones de normalización que se tomarán más adelante en el preprocesamiento; por otro lado, detectar anomalías, comportamientos atípicos o patrones que ameriten una investigación más profunda antes del modelado.

Para cada una de las cinco variables numéricas de interés —excluyendo los identificadores ya descartados— se calculan la media, la mediana y la moda como indicadores de tendencia central, y el mínimo, el máximo, el rango, la varianza, el desvío estándar y el coeficiente de variación como métricas de dispersión. Posteriormente se enriquece este resumen con el rango intercuartílico (IQR), el coeficiente de asimetría y la curtosis, conformando así un perfil estadístico completo de cada variable.

| Variable | Media | Desvío | Asimetría | Curtosis | Observación |
|----------|-------|--------|-----------|----------|-------------|
| `air_temp [K]` | 300.48 | 1.95 | −0.22 | −0.94 | Distribución aproximadamente simétrica, ligeramente achatada |
| `process_temp [K]` | 310.17 | 1.33 | −0.22 | −0.24 | Similar a `air_temp`, con leve sesgo negativo |
| `speed [RPM]` | 1513.39 | 313.60 | **2.08** | **7.99** | Fuertemente asimétrica a la derecha, leptocúrtica |
| `torque [Nm]` | 44.96 | 14.23 | −0.52 | −0.02 | Sesgo negativo moderado |
| `tool_wear [min]` | 124.92 | 70.11 | −0.17 | −1.30 | Distribución achatada (platicúrtica) |

Las temperaturas —tanto la del aire como la del proceso— exhiben distribuciones notablemente simétricas y concentradas en torno a sus medias. Con desvíos estándar de apenas 1.95 K y 1.33 K respectivamente y coeficientes de variación inferiores al 1 %, estas variables presentan muy poca dispersión relativa, lo cual es esperable en procesos industriales controlados donde las condiciones térmicas se mantienen dentro de rangos operativos estrechos. La leve asimetría negativa (−0.22 en ambos casos) indica una cola izquierda ligeramente más pesada, sugiriendo que las excursiones hacia temperaturas inferiores a la media son marginalmente más frecuentes que las excursiones hacia valores superiores.

La variable `speed [RPM]` merece una atención especial. Su coeficiente de asimetría de 2.08 y su curtosis de 7.99 revelan una distribución fuertemente asimétrica hacia la derecha y de colas extremadamente pesadas (leptocúrtica). Esto significa que, si bien la mayoría de las lecturas se concentran alrededor de los 1350–1560 RPM —como lo confirma el IQR de 207 RPM—, existe una proporción no despreciable de observaciones con velocidades considerablemente superiores, que se extienden hasta un máximo de 2886 RPM. Más preocupante aún es el valor mínimo registrado: **−1 RPM**, una medición físicamente imposible en cualquier sistema rotativo real. Este hallazgo, que afecta a 47 registros, obligará a un tratamiento específico durante la fase de preprocesamiento.

`torque [Nm]` muestra un sesgo negativo moderado (−0.52), lo que indica que los valores de torque tienden a concentrarse en la zona alta de su rango, con una cola más extendida hacia los valores bajos. Por su parte, `tool_wear [min]` presenta una distribución marcadamente achatada (curtosis de −1.30), que refleja una dispersión casi uniforme del desgaste a lo largo de todo el intervalo posible, desde 0 hasta 253 minutos. Este comportamiento es coherente con la naturaleza acumulativa del desgaste en un contexto de producción continua: las herramientas se reemplazan periódicamente, por lo que el desgaste observado en cualquier momento dado tiende a distribuirse de manera aproximadamente uniforme en el intervalo de vida útil.

### 3.2 Análisis de distribuciones

Para complementar el resumen numérico con una comprensión visual de la forma de cada variable, el notebook genera dos grillas de gráficos cuidadosamente dispuestas en una cuadrícula de filas y columnas calculada dinámicamente. La primera grilla contiene histogramas con estimación de densidad kernel (KDE) superpuesta, lo que permite apreciar simultáneamente la frecuencia empírica de los valores y una estimación suavizada de la función de densidad subyacente. La segunda grilla despliega diagramas de caja (boxplots), que condensan en una sola figura la mediana, los cuartiles, el rango intercuartílico y los valores atípicos según el criterio de Tukey.

**Imagen de referencia:** Histogramas con KDE (primera figura de la celda `61c66f9c`) y Boxplots (segunda figura de la misma celda).

La inspección visual confirma y enriquece las conclusiones del análisis numérico. Los histogramas de `air_temp [K]` y `process_temp [K]` revelan campanas aproximadamente gaussianas, centradas respectivamente en torno a 300 K y 310 K, con una dispersión acotada que refleja la estabilidad del proceso térmico. La curva KDE se adhiere con fidelidad a la forma del histograma, reforzando la impresión de normalidad aproximada de estas mediciones.

El histograma de `speed [RPM]` es, con diferencia, el más revelador de todos. La concentración masiva de observaciones en el rango de 1200 a 1700 RPM contrasta con una cola derecha que se desvanece lentamente hasta rozar las 2900 RPM. Los boxplots confirman esta estructura: la caja central es compacta, pero los bigotes se extienden generosamente hacia la derecha, y una nube de puntos individuales marca la presencia de outliers más allá del límite superior. La decisión de no aplicar clipping a esta variable durante el preprocesamiento —documentada más adelante— se origina precisamente en esta evidencia visual: los valores elevados no son errores de medición sino regímenes operativos reales, aunque poco frecuentes, del equipamiento.

El diagrama de caja de `torque [Nm]` muestra una leve asimetría negativa, con una caja más densa en la región superior y algunos valores atípicos inferiores que se alejan del bigote bajo. `tool_wear [min]`, por su parte, produce un boxplot notablemente simétrico y extendido, con una caja que abarca prácticamente la mitad del rango total y bigotes que alcanzan los extremos sin señalar outliers, consistente con su distribución achatada.

### 3.3 Proporciones de variables categóricas

El análisis de las dos variables categóricas del dataset —`product_type` y `target`— se aborda mediante el cálculo de frecuencias relativas y su representación gráfica en diagramas de barras.

**Imagen de referencia:** Gráficos de barras de frecuencia para `product_type` y `target` (celda `25a9b4f1`).

| Categoría | Porcentaje |
|-----------|-----------|
| `product_type` L | 71.36 % |
| `product_type` M | 18.53 % |
| `product_type` H | 10.11 % |
| `target` failure | 51.53 % |
| `target` normal | 48.47 % |

La variable `product_type` exhibe un desbalance pronunciado: más del 71 % de las observaciones corresponden al tipo L, mientras que el tipo M apenas supera el 18 % y el tipo H roza el 10 %. Esta distribución desigual tiene implicancias relevantes para el modelado. Por un lado, convierte a `product_type` en una variable potencialmente muy informativa —si los tipos de producto se asocian de manera diferenciada con las fallas— pero, por otro lado, introduce el riesgo de que los modelos aprendan a predecir basándose en la frecuencia base de cada tipo en lugar de en las verdaderas relaciones causales con la variable objetivo. Esta tensión se explorará más adelante mediante la tabla cruzada entre `product_type` y `target`.

La variable objetivo, en cambio, presenta una distribución notablemente equilibrada: un 51.53 % de las observaciones corresponden a fallas y un 48.47 % a operación normal. Esta cercanía al balance perfecto es una excelente noticia desde la perspectiva del modelado, ya que elimina la necesidad de recurrir a técnicas de oversampling sintético como SMOTE —que introducirían datos artificiales con el consiguiente riesgo de sobreajuste— y permite que incluso un undersampling conservador, como el que efectivamente se aplica más adelante, sea suficiente para alcanzar una distribución 50/50 sin pérdida significativa de información.

### 3.4 Relaciones entre variables y conclusiones del EDA

La última etapa del análisis exploratorio se consagra al examen de las relaciones entre variables, tanto desde una perspectiva incondicional —mediante la matriz de correlación lineal— como condicionada por la clase objetivo —mediante scatterplots y boxplots estratificados. Esta es, posiblemente, la sección más rica del EDA, porque las relaciones entre variables son precisamente lo que los modelos de clasificación explotarán para trazar la frontera de decisión entre falla y normalidad.

**Imágenes de referencia:**
- Matriz de correlación (heatmap, primera figura de la celda `62ff3253`).
- Scatter `speed [RPM]` vs `torque [Nm]` coloreado por `target` (segunda figura de la misma celda).
- Scatter `air_temp [K]` vs `process_temp [K]` coloreado por `target` (tercera figura).
- Boxplot `tool_wear [min]` estratificado por `target` (cuarta figura).

La matriz de correlación revela dos relaciones lineales dominantes que estructuran el espacio de features. En primer lugar, `air_temp [K]` y `process_temp [K]` exhiben una **correlación positiva de +0.86**, un valor que desde una perspectiva puramente estadística podría justificar la eliminación de una de las dos variables para reducir multicolinealidad. Sin embargo, como se argumenta en las notas finales del notebook, esta correlación tiene una interpretación física directa: el aire del entorno y el fluido del proceso intercambian calor de manera continua, por lo que es natural que sus temperaturas co-evolucionen. La diferencia entre ambas —aproximadamente 10 K en promedio— refleja el gradiente térmico que impulsa ese intercambio, y es precisamente esa diferencia la que podría contener información predictiva sobre el estado de la máquina.

En segundo lugar, `speed [RPM]` y `torque [Nm]` muestran una **correlación negativa de −0.85**. Esta relación es una consecuencia directa de la ley de potencia en sistemas rotativos: para una misma potencia entregada, un aumento en la velocidad de rotación implica necesariamente una disminución del torque, y viceversa. El scatterplot de estas dos variables, coloreado por la clase objetivo, es particularmente revelador: las observaciones de falla tienden a concentrarse en la región de alta velocidad y bajo torque (esquina inferior derecha del gráfico), mientras que las operaciones normales ocupan predominantemente la región de baja velocidad y alto torque. Esta separación, aunque no perfecta, sugiere que la combinación velocidad–torque es uno de los predictores más poderosos de la condición de la máquina.

El análisis de `tool_wear [min]` estratificado por `target` mediante un boxplot confirma una intuición fundamental en mantenimiento predictivo: el desgaste acumulado de la herramienta es sistemáticamente mayor en las observaciones clasificadas como falla. La mediana de desgaste en el grupo `failure` supera a la del grupo `normal` por un margen considerable, y la dispersión en el grupo de fallas es también más amplia, lo que sugiere que el desgaste elevado es una condición necesaria pero no suficiente para la ocurrencia de fallas —existen observaciones con desgaste alto que no derivaron en falla, presumiblemente porque otras variables operativas se mantuvieron dentro de rangos seguros.

La tabla cruzada entre `product_type` y `target` añade una capa adicional de complejidad al panorama. El producto tipo **L** está fuertemente sobrerrepresentado en las fallas: constituye el 82.9 % de los casos de falla frente a solo el 59.1 % de los casos normales. En contraste, el producto tipo **M** está drásticamente subrepresentado en fallas (7.3 % frente a 30.4 % en normales). Esta asimetría convierte a `product_type` en una variable de alto valor predictivo, pero también en un potencial confusor: si el tipo de producto está correlacionado tanto con las condiciones operativas como con la probabilidad de falla, los modelos podrían aprender asociaciones espurias que no se generalicen a escenarios donde la distribución de tipos de producto sea diferente.

**Conclusión del EDA (reproducida del notebook):**

> - La variable `target` está relativamente balanceada: ~51.53 % `failure` y ~48.47 % `normal`.
> - `product_type` está desbalanceada a favor de `L` (~71.36 %), seguida de `M` y `H`.
> - Hay una correlación positiva alta entre `air_temp [K]` y `process_temp [K]` (~0.86).
> - Hay una correlación negativa fuerte entre `speed [RPM]` y `torque [Nm]` (~-0.82).
> - El `tool_wear [min]` tiende a ser mayor en observaciones con `failure` que en `normal`.
> - Se observan valores atípicos en `speed [RPM]` y en menor medida en otras variables, visibles en los boxplots.

Cabe señalar que el valor de correlación speed–torque citado en esta conclusión (~-0.82) difiere ligeramente del que arroja la matriz de correlación del EDA (−0.85). La diferencia es pequeña y no altera la interpretación sustantiva, pero constituye una inconsistencia menor que se documenta en la sección de errores de este reporte.

---

## 4. Fase 3 — Preprocesamiento para modelado

Concluido el análisis exploratorio, el notebook ingresa en la fase de preprocesamiento, donde las observaciones cualitativas del EDA se traducen en transformaciones concretas sobre los datos. Esta fase es particularmente crítica porque las decisiones que se toman aquí —cómo imputar valores faltantes, qué variables conservar o eliminar, cómo tratar los outliers, cómo codificar las categóricas y cómo normalizar— condicionan de manera irreversible la calidad de los modelos que se entrenarán posteriormente. El preprocesamiento se articula en cuatro etapas secuenciales.

### 4.1 Limpieza inicial e imputación con KMeans

La primera etapa del preprocesamiento aborda tres problemas detectados durante el EDA: la presencia de columnas irrelevantes, las filas duplicadas y los valores numéricos inconsistentes o faltantes. La secuencia de operaciones es la siguiente:

1. Se eliminan las columnas identificadoras `idx` y `parent_device_id`, que habían sido apartadas del análisis numérico pero aún residían en el DataFrame. Su eliminación es definitiva para el modelado.

2. Se detectan y eliminan **72 filas duplicadas**, es decir, registros idénticos en todas sus columnas. Esta operación es estándar en cualquier pipeline de datos, ya que los duplicados pueden inflar artificialmente ciertas regiones del espacio de features y distorsionar tanto las estimaciones de rendimiento durante la validación cruzada como las predicciones del modelo final.

3. Se identifican **47 registros** donde `speed [RPM]` presenta valores menores o iguales a cero, una imposibilidad física para un sistema rotativo. Estos valores se reemplazan por `NaN` para ser tratados como datos faltantes en el paso siguiente, en lugar de ser eliminados —lo que preserva el resto de la información de esas filas (temperaturas, torque, desgaste y clase objetivo).

4. Se implementa una función de imputación personalizada, `impute_with_kmeans()`, que constituye uno de los aspectos metodológicamente más sofisticados del trabajo. En lugar de recurrir a la imputación univariada tradicional (media, mediana o moda), esta función aprovecha la alta correlación entre pares de variables para estimar los valores faltantes a partir de la estructura bivariada de los datos. El procedimiento es el siguiente:
   - Para cada par de variables altamente correlacionadas —`(torque, speed)` con correlación ~−0.85 y `(process_temp, air_temp)` con correlación ~0.86— se entrena un modelo KMeans sobre los registros completos del par.
   - Para cada observación con el valor faltante, se identifica el centroide del clúster más cercano en la dimensión de la variable predictora (la que sí está presente).
   - El valor imputado es la coordenada de ese centroide en la dimensión de la variable a completar.

   Este enfoque tiene una virtud fundamental: respeta la relación funcional entre las variables. En el caso de `speed` y `torque`, donde la física dicta que a mayor velocidad corresponde menor torque para una misma potencia, imputar con la media global de velocidad ignoraría completamente esta dependencia y podría introducir inconsistencias —por ejemplo, asignar una velocidad baja a una observación con torque bajo, cuando la relación real esperable sería una velocidad alta. KMeans, al agrupar las observaciones completas y luego proyectar sobre los centroides, preserva la coherencia bivariada.

5. Los valores nulos remanentes —aquellos que no pertenecen a ninguno de los pares correlacionados o que no pudieron ser imputados en el paso anterior— se completan con la mediana para las variables numéricas y con la moda para las variables categóricas, como estrategia de respaldo.

**Imagen de referencia:** Scatter post-imputación de `speed [RPM]` vs `torque [Nm]` (celda `speed_torque_scatter_after`), que permite verificar visualmente que los puntos imputados se integran de manera natural en la estructura de correlación preexistente, sin introducir artefactos visibles.

### 4.2 Evaluación de multicolinealidad

Con los datos ya limpios y completos, el notebook aborda la cuestión de la multicolinealidad. El procedimiento consiste en calcular la matriz de correlación (en valor absoluto) sobre las variables numéricas preprocesadas, y luego iterar sobre todos los pares de variables para identificar aquellos cuya correlación supere un umbral predefinido. La variable que presente mayor correlación promedio con el resto de las features es la candidata a ser eliminada, bajo el criterio de que su información está más redundantemente representada en el conjunto.

El umbral elegido es 0.88, un valor más permisivo que los típicos 0.80 o 0.85 que suelen emplearse en la práctica. Esta elección no es arbitraria: refleja la decisión consciente del equipo de ser conservador en la eliminación de variables, priorizando la preservación de información potencialmente útil siempre que la redundancia no sea extrema.

El resultado de este análisis es que **ningún par supera el umbral de 0.88**. Las correlaciones post-imputación son 0.860 para el par `air_temp`–`process_temp` y 0.852 (en valor absoluto) para el par `speed`–`torque`. En consecuencia, no se elimina ninguna variable, y las cinco features numéricas originales se conservan íntegramente para el modelado. Esta decisión es posteriormente validada en las notas finales mediante un análisis de eliminación recursiva de features (RFE manual) que confirma que la inclusión de ambas temperaturas mejora o mantiene el rendimiento de los clasificadores.

### 4.3 Tratamiento de valores atípicos

El tratamiento de outliers se implementa siguiendo el método del rango intercuartílico (IQR), posiblemente la técnica más extendida para la detección no paramétrica de valores atípicos. Para cada variable numérica, se calculan el primer cuartil (Q1) y el tercer cuartil (Q3), y se define como outlier cualquier valor que se encuentre por debajo de Q1 − 1.5 × IQR o por encima de Q3 + 1.5 × IQR.

La detección arroja los siguientes resultados:

| Variable | Outliers | % del dataset |
|----------|----------|---------------|
| `speed [RPM]` | 1.128 | 7.81 % |
| `torque [Nm]` | 106 | 0.73 % |
| `process_temp [K]` | 67 | 0.46 % |
| `air_temp [K]` | 0 | 0.00 % |
| `tool_wear [min]` | 0 | 0.00 % |

La estrategia de tratamiento diferencia dos casos. Para `torque [Nm]`, `process_temp [K]`, `air_temp [K]` y `tool_wear [min]`, se aplica **clipping**: los valores que exceden los límites IQR son truncados al valor del límite correspondiente. Esta técnica tiene la ventaja de mitigar el impacto de los valores extremos —que de otro modo podrían distorsionar los coeficientes de la regresión logística o las distancias en KNN— sin descartar las observaciones completas, preservando así el tamaño de la muestra y la información contenida en las demás variables de esas filas.

Para `speed [RPM]`, en cambio, se **omite deliberadamente el clipping**. La justificación, explicitada en las notas finales, es que los valores elevados de velocidad no son errores de medición sino regímenes operativos reales del equipamiento, y eliminarlos o truncarlos equivaldría a descartar información legítima sobre el comportamiento de la máquina en condiciones extremas —precisamente las condiciones donde las fallas son más probables y, por tanto, más informativas para el modelo. Se trata de una decisión consciente y con fundamento de dominio, aunque no exenta de riesgo: los 1.128 outliers (7.81 % del dataset) permanecen en los datos y podrían afectar el rendimiento de modelos sensibles a la escala y a las distancias, particularmente KNN y la regresión logística.

### 4.4 Codificación, balanceo de clases y normalización

La etapa final del preprocesamiento transforma el dataset en un formato completamente numérico y balanceado, listo para ser consumido por cualquier algoritmo de clasificación. Las operaciones se ejecutan en cuatro pasos:

1. **One-hot encoding de `product_type`.** Dado que esta variable tiene tres categorías (L, M, H), se generan dos columnas binarias mediante `pd.get_dummies()` con el parámetro `drop_first=True`. Esto produce las columnas `product_type_L` y `product_type_M`; la categoría H queda implícitamente codificada como el caso en que ambas columnas son `False`. Esta representación evita la trampa de la multicolinealidad perfecta que introducirían tres columnas dummy para tres categorías.

2. **Binarización de la variable objetivo.** La columna `target`, originalmente con valores `'failure'` y `'normal'`, se mapea a 1 y 0 respectivamente, convirtiendo el problema en una clasificación binaria estándar.

3. **Balanceo de clases con RandomUnderSampler.** Aunque las clases estaban cerca del equilibrio (~51.5 % vs. ~48.5 %), se aplica un submuestreo aleatorio de la clase mayoritaria (`failure`) para igualarla exactamente a la minoritaria (`normal`). El resultado es un dataset perfectamente balanceado de 14.078 observaciones (7.039 por clase). La elección de undersampling por sobre oversampling sintético (SMOTE) es deliberada y se fundamenta en el bajo nivel de desbalance original: con una diferencia de apenas 3 puntos porcentuales, la pérdida de información por submuestreo es mínima (~371 filas eliminadas), mientras que la generación de datos sintéticos con SMOTE introduciría un riesgo innecesario de sobreajuste.

4. **Normalización de variables numéricas.** Este paso, que en la versión original del notebook contenía un error crítico ya corregido (ver sección 7.1), aplica `StandardScaler()` de scikit-learn para transformar cada variable numérica a una distribución con media 0 y desvío estándar 1. La estandarización z-score es esencial para modelos sensibles a la escala como la regresión logística (cuyos coeficientes de regularización dependen de la magnitud de las features), KNN (donde las distancias euclidianas o de Manhattan se verían dominadas por las variables de mayor varianza) y, en menor medida, para las redes neuronales y las máquinas de soporte vectorial. Los modelos basados en árboles, en cambio, son invariantes a transformaciones monótonas de las features, por lo que la estandarización no los afecta.

**Resultado final del preprocesamiento:**

| Propiedad | Valor |
|-----------|-------|
| Shape de X | (14.078, 7) |
| Shape de y | (14.078,) |
| Balance de clases | 50 % / 50 % |
| Distribución de product_type en test | L: 70.4 %, M: 19.7 %, H: 9.9 % |
| Columnas | 5 numéricas estandarizadas + 2 one-hot de product_type |

---

## 5. Fase 4 — Modelado y evaluación de clasificadores

La fase de modelado constituye la culminación técnica del trabajo. Sobre los datos preprocesados, se entrenan seis clasificadores pertenecientes a familias algorítmicas distintas, se optimizan sus hiperparámetros mediante búsqueda exhaustiva con validación cruzada, y se evalúa su rendimiento sobre un conjunto de prueba independiente utilizando un abanico de métricas complementarias. El objetivo no es simplemente identificar el modelo con mejor desempeño, sino comprender cómo cada familia algorítmica se relaciona con la estructura subyacente de los datos —si la frontera de decisión es aproximadamente lineal, si las features son condicionalmente independientes, si la información relevante reside en la vecindad local de cada punto o si es necesario combinar múltiples perspectivas mediante ensambles.

### 5.1 División entrenamiento-prueba

El primer paso consiste en separar el dataset balanceado en dos conjuntos disjuntos: uno de entrenamiento, que contendrá el 70 % de las observaciones (9.854 instancias), y uno de prueba con el 30 % restante (4.224 instancias). La división se realiza con `train_test_split` utilizando el parámetro `stratify=y_bal`, lo que garantiza que la proporción de clases se mantenga exactamente en 50/50 tanto en entrenamiento como en prueba. La semilla aleatoria se fija en 42 para asegurar la reproducibilidad completa del experimento.

Como producto adicional de esta etapa, los conjuntos de entrenamiento y prueba se exportan a los archivos `train_data.csv` y `test_data.csv`, lo que facilita la reutilización de los datos procesados en futuras iteraciones del proyecto sin necesidad de re-ejecutar todo el pipeline de preprocesamiento. Adicionalmente, se verifica la distribución de `product_type` en el conjunto de prueba, confirmando que la proporción de los tipos L, M y H (70.4 %, 19.7 % y 9.9 % respectivamente) es consistente con la distribución observada en el dataset original.

### 5.2 Estrategia de validación cruzada

Para la búsqueda de hiperparámetros de todos los modelos, se adopta una estrategia uniforme de validación cruzada estratificada con 5 folds (`StratifiedKFold`, `n_splits=5`, `shuffle=True`, `random_state=42`). La estratificación asegura que cada fold preserve la distribución de clases del conjunto de entrenamiento, evitando que folds desbalanceados introduzcan varianza espuria en las estimaciones de rendimiento. La métrica utilizada como criterio de optimización en `GridSearchCV` es el **F1-score**, que equilibra precision y recall y resulta particularmente adecuada para problemas de clasificación binaria donde tanto los falsos positivos como los falsos negativos tienen costo.

### 5.3 Modelos entrenados y búsqueda de hiperparámetros

Se entrenan y optimizan seis modelos, cada uno con su propia grilla de hiperparámetros:

| # | Modelo | Hiperparámetros explorados | Mejor configuración | F1 en CV |
|---|--------|---------------------------|---------------------|----------|
| 1 | **Regresión Logística** | `C` ∈ {0.01, 0.1, 1, 10, 100}, `solver` ∈ {lbfgs, liblinear} | C = 0.1, solver = liblinear | 0.8452 |
| 2 | **Naive Bayes (GaussianNB)** | `var_smoothing` ∈ [1e-12, 1e-6] (7 valores log-espaciados) | var_smoothing = 1e-12 | 0.8130 |
| 3 | **KNN** | `n_neighbors` ∈ {3, 5, 7, 9, 11, 15}, `weights` ∈ {uniform, distance}, `metric` ∈ {euclidean, manhattan} | k = 3, weights = distance, metric = manhattan | 0.9516 |
| 4 | **Árbol de Decisión** | `max_depth` ∈ {3, 5, 7, 10, None}, `min_samples_split` ∈ {2, 5, 10}, `min_samples_leaf` ∈ {1, 2, 4}, `criterion` ∈ {gini, entropy} | criterion = entropy, max_depth = None, min_samples_leaf = 1, min_samples_split = 2 | 0.9526 |
| 5 | **Random Forest** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {5, 10, None}, `min_samples_split` ∈ {2, 5}, `min_samples_leaf` ∈ {1, 2}, `criterion` ∈ {gini, entropy} | n_estimators = 100, criterion = entropy, max_depth = None, min_samples_leaf = 1, min_samples_split = 2 | 0.9660 |
| 6 | **Gradient Boosting** | `n_estimators` ∈ {100, 200}, `learning_rate` ∈ {0.05, 0.1, 0.2}, `max_depth` ∈ {3, 5}, `min_samples_split` ∈ {2, 5} | n_estimators = 200, learning_rate = 0.2, max_depth = 5, min_samples_split = 5 | 0.9716 |

Varios patrones merecen ser señalados en estos resultados de validación cruzada. En primer lugar, se observa un claro agrupamiento por familia algorítmica: los modelos lineales (LR y NB) obtienen F1 en el rango 0.81–0.85, los modelos basados en instancias y árboles simples (KNN y DT) se sitúan en torno a 0.95, y los ensembles (RF y GB) superan el 0.96. Esta jerarquía es exactamente la esperable para un problema donde las relaciones entre features y la variable objetivo no son perfectamente lineales ni las features son condicionalmente independientes.

En segundo lugar, los hiperparámetros óptimos revelan características interesantes de los datos. El hecho de que KNN prefiera `k=3` y `weights='distance'` sugiere que la información relevante para la clasificación está codificada en la vecindad muy local de cada punto, y que los vecinos más cercanos son desproporcionadamente más informativos que los lejanos. La preferencia del Árbol de Decisión por `max_depth=None` y `min_samples_leaf=1` indica que el modelo se beneficia de una granularidad muy fina en las particiones, sin riesgo inminente de sobreajuste en validación cruzada. Para Random Forest, la selección de 100 árboles (en lugar de 200) con `max_depth=None` sugiere que 100 estimadores son suficientes para estabilizar el ensemble, y que la profundidad ilimitada no incurre en sobreajuste gracias al promediado.

### 5.4 Resultados sobre el conjunto de prueba

Una vez seleccionada la mejor configuración de hiperparámetros para cada modelo mediante validación cruzada, se evalúa su rendimiento sobre el conjunto de prueba —datos que ningún modelo ha visto durante el entrenamiento ni durante la selección de hiperparámetros— utilizando seis métricas complementarias:

| Métrica | Reg. Logística | Naive Bayes | KNN | Árbol Dec. | Random Forest | Grad. Boosting |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Accuracy** | 0.8364 | 0.8092 | 0.9531 | 0.9545 | 0.9697 | **0.9777** |
| **Precision** | 0.8306 | 0.7644 | 0.9223 | 0.9457 | 0.9550 | **0.9676** |
| **Recall** | 0.8452 | 0.8939 | **0.9896** | 0.9645 | 0.9858 | 0.9886 |
| **F1-Score** | 0.8378 | 0.8241 | 0.9548 | 0.9550 | 0.9702 | **0.9780** |
| **ROC AUC** | 0.9197 | 0.8904 | 0.9785 | 0.9545 | 0.9966 | **0.9973** |

Además de estas métricas agregadas, el notebook genera para cada modelo su correspondiente **matriz de confusión** y, para el conjunto completo, un gráfico de **curvas ROC superpuestas** que permite comparar visualmente la capacidad discriminativa de todos los clasificadores en un mismo espacio.

**Imágenes de referencia:**
- Matrices de confusión de los seis modelos (celdas `cm_lr`, `cm_nb`, `cm_knn`, `cm_dt`, `cm_rf`, `cm_gb`).
- Curvas ROC comparativas (celda `tabla_comparativa`, segunda figura).
- Tabla comparativa de métricas (celda `tabla_comparativa`, DataFrame).

### 5.5 Análisis comparativo del rendimiento

Los resultados sobre el conjunto de prueba confirman y, en algunos casos, intensifican la jerarquía observada durante la validación cruzada. El ranking final, ordenado por F1-score, es el siguiente:

1. 🥇 **Gradient Boosting**: F1 = 0.9780, AUC = 0.9973. El modelo con mejor rendimiento global. Su capacidad discriminativa es prácticamente perfecta: un AUC de 0.9973 significa que, en el 99.73 % de los pares aleatorios (falla, normal), el modelo asigna una probabilidad más alta a la observación de falla. La precisión de 0.9676 y el recall de 0.9886 indican que el modelo es ligeramente más conservador al predecir fallas (privilegia evitar falsos positivos), pero aún así captura casi la totalidad de las fallas reales.

2. 🥈 **Random Forest**: F1 = 0.9702, AUC = 0.9966. A una distancia de menos de un punto porcentual de GB en todas las métricas, Random Forest ofrece un rendimiento virtualmente equivalente. La diferencia práctica entre ambos es minúscula, lo que convierte la elección entre ellos en una cuestión de preferencias operativas: RF es inherentemente paralelizable (cada árbol se entrena de forma independiente), mientras que GB construye los árboles secuencialmente.

3. 🥉 **Árbol de Decisión** (F1 = 0.9550) y **KNN** (F1 = 0.9548): Resulta sorprendente —y testimonio de la calidad de las features— que un único árbol de decisión sin límite de profundidad alcance un rendimiento tan cercano al de los ensembles, superando incluso a KNN por una fracción mínima. El Árbol de Decisión logra un equilibrio notable entre precisión (0.9457) y recall (0.9645). KNN, por su parte, exhibe el **recall más alto de todos los modelos** (0.9896), lo que lo convierte en la opción preferible si el costo de un falso negativo (falla no detectada) es mucho mayor que el de un falso positivo (falsa alarma).

4. **Regresión Logística**: F1 = 0.8378, AUC = 0.9197. Tras la corrección de la normalización, la regresión logística muestra una mejora sustancial respecto de los valores originales (F1 pasó de 0.79 a 0.84). Sin embargo, el gap de aproximadamente 12 puntos de F1 respecto a los modelos de árboles confirma que la frontera de decisión entre falla y normalidad no es satisfactoriamente aproximable por un hiperplano en el espacio de features original. La incorporación de términos de interacción o transformaciones no lineales de las features podría reducir este gap, pero a costa de sacrificar la interpretabilidad que es la principal fortaleza de la regresión logística.

5. **Naive Bayes**: F1 = 0.8241, AUC = 0.8904. Es el modelo con el rendimiento más bajo, lo cual es consistente con la violación de su supuesto fundamental: la independencia condicional de las features. Con correlaciones de 0.86 entre las temperaturas y −0.85 entre velocidad y torque, el supuesto de independencia se vulnera de manera significativa. Con todo, el rendimiento no es despreciable (F1 > 0.82, AUC > 0.89), lo que habla de la robustez de GaussianNB incluso cuando sus supuestos teóricos no se cumplen estrictamente. Cabe señalar que, con la normalización incorrecta anterior (`Normalizer`), Naive Bayes alcanzaba un F1 artificialmente inflado de 0.85, debido a que la proyección sobre la hiperesfera unitaria reducía las correlaciones entre features y acercaba los datos a la condición de independencia. Al corregir a `StandardScaler`, el modelo recibe las features en su escala real y su rendimiento refleja de manera más honesta la adecuación de sus supuestos a la estructura de los datos.

---

## 6. Fase 5 — Notas finales y fundamentación de decisiones

El notebook concluye con una celda de markdown que documenta y fundamenta cuatro decisiones metodológicas clave adoptadas durante el desarrollo del proyecto. Esta sección es particularmente valiosa porque transforma lo que de otro modo serían elecciones arbitrarias de preprocesamiento en decisiones razonadas, transparentes y auditables.

### 6.1 Preservación de ambas variables de temperatura

La decisión de conservar tanto `air_temp [K]` como `process_temp [K]` a pesar de su alta correlación positiva (+0.86) se fundamenta en dos pilares complementarios. El primero es de naturaleza física: la temperatura del aire y la temperatura del proceso representan magnitudes termodinámicamente distintas. La primera refleja las condiciones ambientales o de refrigeración del entorno de la máquina, mientras que la segunda captura el estado térmico interno del proceso de manufactura. La diferencia entre ambas —el gradiente térmico— es una magnitud con significado físico propio que podría desvanecerse si se eliminara una de las variables.

El segundo pilar es de naturaleza empírica: el equipo realizó una validación mediante **eliminación recursiva manual de features** (RFE manual), entrenando los modelos con y sin cada una de las temperaturas y comparando las métricas resultantes. Los resultados de este experimento —que no se incluyen explícitamente en el notebook pero cuya realización se documenta— mostraron que la inclusión de ambas temperaturas mejora o, en el peor de los casos, mantiene el rendimiento de los clasificadores. Esta validación empírica es exactamente el tipo de evidencia que debería respaldar la decisión de conservar o eliminar una variable, en contraposición a la práctica mecánica de eliminar features basándose exclusivamente en umbrales de correlación.

### 6.2 Undersampling como estrategia de balanceo

Con una diferencia de apenas 3 puntos porcentuales entre las clases (51.5 % failure vs. 48.5 % normal), el equipo descartó el uso de técnicas de oversampling sintético como SMOTE o ADASYN. La decisión se apoya en un principio de parsimonia: si el desbalance es mínimo, la generación de observaciones artificiales introduce un riesgo de sobreajuste —el modelo podría aprender patrones espurios de las muestras sintéticas— que no se justifica por la ganancia marginal en balance. El `RandomUnderSampler`, al eliminar aleatoriamente una pequeña fracción de la clase mayoritaria (~371 de ~7.400 observaciones), logra un balance perfecto con una pérdida de información insignificante y sin contaminar el dataset con datos sintéticos.

### 6.3 Imputación bivariada con KMeans para pares correlacionados

Esta es, probablemente, la decisión metodológica más sofisticada del trabajo. La función `impute_with_kmeans()` implementa una estrategia de imputación que trasciende el enfoque univariado tradicional. Al entrenar KMeans sobre los registros completos de cada par correlacionado y luego proyectar las observaciones incompletas sobre los centroides, la imputación preserva la estructura de dependencia bivariada. En el caso concreto de `speed [RPM]` y `torque [Nm]`, donde la física dicta una relación inversa gobernada por la ley de potencia, esta estrategia asegura que los valores imputados sean físicamente coherentes: una observación con torque alto recibirá una velocidad baja imputada, y viceversa, en lugar de recibir la media global de velocidad que podría ser inconsistentemente alta o baja para ese nivel de torque.

### 6.4 Resumen de la pipeline completa

El notebook sintetiza la pipeline en siete pasos, proporcionando una visión panorámica del flujo de trabajo completo:

1. Limpieza de identificadores (`idx`, `parent_device_id`) y eliminación de filas duplicadas.
2. Corrección de valores inválidos de `speed [RPM]` (RPM ≤ 0) e imputación conjunta de pares correlacionados mediante KMeans.
3. Eliminación de variables altamente correlacionadas solo cuando el umbral lo justifica y la validación empírica no muestra pérdida de rendimiento (en este caso, no se eliminó ninguna variable).
4. Estandarización z-score de las variables numéricas con `StandardScaler`.
5. One-hot encoding de la variable categórica `product_type`.
6. División estratificada en conjuntos de entrenamiento (70 %) y prueba (30 %).
7. Entrenamiento y optimización de hiperparámetros con `GridSearchCV` sobre seis modelos de clasificación, seguido de evaluación multimétrica sobre el conjunto de prueba.

---

## 7. Errores detectados y correcciones aplicadas

Durante la revisión del notebook se identificaron cuatro incidencias, de las cuales dos eran errores de código con impacto material en los resultados y dos eran discrepancias cosméticas o artefactos de ejecución. Las dos incidencias con impacto han sido corregidas y el notebook re-ejecutado en su totalidad. A continuación se documentan todas ellas con el detalle necesario para comprender su naturaleza, su impacto y la corrección aplicada.

### 7.1 Error en la normalización: `Normalizer()` en lugar de `StandardScaler()`

**Ubicación:** Celda `76e62fae` (Parte 2, «Procesamiento de categóricas, balance y normalización»).

Este es, con diferencia, el hallazgo más significativo de la revisión. En la etapa de normalización de las variables numéricas, el código original utilizaba:

```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
```

El comentario que acompañaba a esta línea decía «Normalización estándar (z-score)», pero la clase `Normalizer` de scikit-learn **no implementa estandarización z-score**. `Normalizer` con sus parámetros por defecto (`norm='l2'`) escala cada **fila** del dataset para que tenga norma euclidiana unitaria, proyectando todas las muestras sobre la superficie de la hiperesfera unidad. La estandarización z-score, en cambio, opera por **columna**: centra cada variable en media 0 y la escala a desvío estándar 1. La diferencia es fundamental y tiene consecuencias profundas sobre el comportamiento de los modelos.

El impacto de esta confusión es triple. En primer lugar, la normalización L2 por fila distorsiona las distancias entre muestras, lo que afecta directamente a KNN (cuyas predicciones dependen exclusivamente de métricas de distancia), a la regresión logística (cuyos coeficientes de regularización L2 dependen de la escala de las features) y, en menor medida, a Naive Bayes. En segundo lugar, la proyección sobre la hiperesfera introduce una dependencia no lineal entre las features de cada muestra —la magnitud de cada feature pasa a estar condicionada por las magnitudes de las demás features de esa misma fila— lo que viola el supuesto de independencia condicional de Naive Bayes pero, paradójicamente, puede beneficiar a GaussianNB cuando las features originales están correlacionadas, al reducir artificialmente esas correlaciones. En tercer lugar, los modelos basados en árboles (Árbol de Decisión, Random Forest, Gradient Boosting) son invariantes a transformaciones monótonas de las features individuales, pero no son invariantes a transformaciones que mezclan features de una misma fila, por lo que incluso ellos podrían verse afectados.

**Corrección aplicada (14/05/2026):**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

**Impacto verificado:** Tras reemplazar `Normalizer` por `StandardScaler` y re-ejecutar el notebook completo, todos los modelos modificaron sus resultados. La Regresión Logística experimentó la mejora más drástica, pasando de un F1 de 0.79 a 0.84 —una ganancia de 5 puntos porcentuales que confirma que el modelo lineal es particularmente sensible a la escala incorrecta de las features. Naive Bayes, en cambio, descendió de 0.85 a 0.82, reflejando que la normalización L2 anterior enmascaraba artificialmente las correlaciones entre features que violan su supuesto de independencia. Los modelos de árboles también mostraron mejoras (DT de 0.94 a 0.96, RF de 0.95 a 0.97, GB de 0.95 a 0.98), lo cual sugiere que las salidas del notebook original no correspondían a una ejecución limpia desde cero sino que incluían resultados stale de ejecuciones parciales previas. El ranking final —GB > RF > DT ≈ KNN > LR > NB— es ahora plenamente consistente con lo que la teoría predice para cada familia algorítmica.

### 7.2 Mensaje de umbral inconsistente en la salida

**Ubicación:** Celda `corr_feature_removal`.

En el código de evaluación de multicolinealidad, la variable `umbral` se define con el valor 0.88, y el bucle itera sobre los pares de variables comparando su correlación contra este umbral. Sin embargo, el mensaje de salida para el caso en que ningún par supera el umbral estaba hardcodeado como:

```python
print('No se detectaron pares con correlacion > 0.9')
```

El valor 0.9 en el string no guarda relación con la variable `umbral = 0.88`. Si en el futuro se modificara el umbral (por ejemplo, a 0.85 o a 0.95), el mensaje de salida seguiría diciendo 0.9, generando una inconsistencia entre lo que el código hace y lo que el output declara. En la ejecución actual, dado que ninguna correlación supera 0.88 (y por tanto tampoco supera 0.9), el mensaje no es factualmente incorrecto, pero sí es engañoso desde el punto de vista del diseño del código.

**Corrección aplicada:**
```python
print(f'No se detectaron pares con correlacion > {umbral}')
```

### 7.3 Discrepancia menor en las conclusiones del EDA

**Ubicación:** Celda markdown `6df2405b` (Conclusiones principales).

La conclusión del EDA afirma que la correlación entre `speed [RPM]` y `torque [Nm]` es «~-0.82». Sin embargo, la matriz de correlación del propio EDA —visualizada en el heatmap de la celda `62ff3253`— muestra un valor de −0.85. La diferencia de 0.03 en valor absoluto no altera la interpretación cualitativa (sigue siendo una correlación negativa fuerte en ambos casos), pero introduce una pequeña inconsistencia entre la afirmación textual y la evidencia numérica presentada. Es posible que el valor ~-0.82 provenga de una versión anterior del notebook donde los datos aún no habían sido depurados (por ejemplo, con los 47 valores de RPM ≤ 0 todavía presentes), lo que explicaría una correlación ligeramente más débil. En cualquier caso, se trata de una discrepancia menor que no afecta las conclusiones sustantivas del trabajo.

### 7.4 Execution counts inconsistentes en el notebook original

**Ubicación:** Metadatos de las celdas de código.

En la versión original del notebook, la primera celda de código mostraba `execution_count: 3` mientras que todas las demás celdas tenían `execution_count: null`. Este patrón es característico de un notebook que fue ejecutado parcialmente —posiblemente solo la primera celda— y cuyo kernel fue posteriormente reiniciado sin volver a ejecutar el resto. Si bien los execution counts no afectan la corrección del código ni la validez de los resultados, su inconsistencia es una señal de alerta sobre la reproducibilidad: si algunas celdas no fueron ejecutadas en orden, las salidas que muestran podrían no corresponder al estado actual del código.

Tras la re-ejecución completa del notebook realizada el 14 de mayo de 2026, las 36 celdas de código presentan execution counts secuenciales del 1 al 36, confirmando una ejecución limpia y ordenada de principio a fin.

---

## 8. Conclusiones generales y recomendaciones

### Sobre el análisis exploratorio

El notebook presenta un EDA meticulosamente estructurado que recorre todas las dimensiones relevantes para un problema de clasificación binaria: caracterización univariada de cada feature, análisis de distribuciones mediante herramientas visuales complementarias (histogramas + KDE + boxplots), examen del balance de clases y las proporciones de variables categóricas, y exploración sistemática de las relaciones bivariadas —tanto incondicionales (correlación) como condicionadas por la clase objetivo (scatterplots y boxplots estratificados). Las visualizaciones son claras, están correctamente etiquetadas y utilizan una paleta de colores consistente a lo largo de todo el trabajo. El único punto de mejora en esta fase sería la inclusión explícita de los valores numéricos de la matriz de correlación en formato tabular —actualmente solo se presentan como heatmap— para facilitar la referencia precisa en el texto de las conclusiones.

### Sobre el preprocesamiento

Las decisiones de preprocesamiento exhiben un equilibrio poco frecuente entre rigor estadístico y conocimiento del dominio. La imputación con KMeans para pares correlacionados trasciende las aproximaciones univariadas estándar y demuestra una comprensión genuina de que los datos industriales están gobernados por leyes físicas que la imputación debe respetar. La decisión de no eliminar ninguna variable por multicolinealidad, respaldada por validación empírica con RFE manual, evita la práctica mecánica de descartar features basándose exclusivamente en umbrales de correlación. La omisión deliberada del clipping para `speed [RPM]`, aunque arriesgada desde una perspectiva puramente estadística, está justificada por el conocimiento de que los valores extremos de velocidad corresponden a regímenes operativos reales y potencialmente informativos.

El error original de `Normalizer()` —ya corregido— sirve como recordatorio de la importancia de verificar que la transformación aplicada coincida con la transformación declarada. La diferencia entre normalización L2 por fila y estandarización z-score por columna no es un detalle menor: es la diferencia entre dos transformaciones que producen espacios de features radicalmente distintos, con consecuencias drásticas sobre el rendimiento relativo de los modelos.

### Sobre el modelado y la evaluación

La selección de seis modelos que abarcan un espectro amplio de familias algorítmicas —desde el modelo lineal más simple hasta ensembles de boosting— permite obtener una caracterización rica de la estructura de los datos. La búsqueda de hiperparámetros con `GridSearchCV` sobre grillas razonablemente densas, combinada con una validación cruzada estratificada de 5 folds, proporciona estimaciones robustas del rendimiento esperado. La evaluación sobre el conjunto de prueba con seis métricas complementarias —accuracy, precision, recall, F1, AUC y matriz de confusión— junto con la visualización de curvas ROC superpuestas, constituye un benchmark completo que no deja puntos ciegos en la comparación de modelos.

Con los datos correctamente escalados, **Gradient Boosting** emerge como el clasificador óptimo para este problema, con un F1 de 0.9780 y un AUC de 0.9973. La cercanía de estos valores a la perfección sugiere dos cosas: primero, que las features disponibles contienen prácticamente toda la información necesaria para predecir la condición de la máquina; segundo, que el margen de mejora restante es muy reducido y probablemente requeriría la incorporación de nuevas variables (por ejemplo, vibración, presión, histórico de mantenimiento) más que ajustes incrementales en los modelos existentes.

### Recomendaciones para trabajo futuro

1. **Validar normalización estandarizada continua.** `StandardScaler` reemplazó adecuadamente a `Normalizer` en las iteraciones finales, lo cual es crítico mantener en despliegues a producción.
2. **Mantener consistencia en la salida de hiperparámetros.** Se corrigió exitosamente la salida que referencia dinámicamente la variable `umbral`.

3. **Implementar validación cruzada anidada (nested CV)**. Actualmente, la selección de hiperparámetros y la estimación del error de generalización comparten el mismo conjunto de prueba, lo que puede producir estimaciones optimistas del rendimiento. Una estructura de nested CV —donde un bucle externo evalúa el error de generalización y un bucle interno selecciona hiperparámetros— proporcionaría estimaciones no sesgadas, particularmente relevante dado que varios modelos alcanzan métricas superiores a 0.97.

4. **Investigar la interacción `product_type` × `target` como posible confusor.** La tabla cruzada revela que el tipo de producto L está drásticamente sobrerrepresentado en las fallas (82.9 % de las fallas vs. 59.1 % de los normales), mientras que el tipo M está subrepresentado (7.3 % vs. 30.4 %). Si esta distribución refleja diferencias reales en la propensión a fallas de cada tipo de producto, `product_type` es una feature legítima. Pero si refleja un sesgo de muestreo —por ejemplo, que las máquinas que fabrican el tipo L fueron monitoreadas con mayor frecuencia durante períodos de falla— entonces `product_type` podría estar actuando como un confusor que los modelos explotan de manera espuria. Un análisis de la Generalización a través de tipos de producto (evaluar el rendimiento del modelo por separado para cada tipo) ayudaría a dilucidar esta cuestión.

5. **Explorar transformaciones no lineales de las features para modelos lineales.** El gap de 14 puntos de F1 entre la Regresión Logística y los ensembles sugiere que la frontera de decisión no es lineal en el espacio de features original. La incorporación de términos polinómicos, interacciones (por ejemplo, `speed × torque`, que tiene una interpretación física como potencia) o transformaciones como el logaritmo de `speed [RPM]` —que reduciría la asimetría de esta variable— podría cerrar parcialmente ese gap sin sacrificar la interpretabilidad que hace valiosa a la regresión logística en contextos industriales donde la explicabilidad es un requisito.

6. **Evaluar la estabilidad temporal del modelo.** Los datos provienen de un proceso industrial donde las condiciones operativas pueden derivar con el tiempo (degradación de componentes, cambios en la materia prima, variaciones estacionales de temperatura ambiente). Un análisis de la estabilidad de las predicciones a lo largo del tiempo —por ejemplo, evaluando el rendimiento en diferentes ventanas temporales si los datos incluyen marcas de tiempo— permitiría anticipar la necesidad de reentrenamiento periódico.

---

*Reporte generado el 14 de mayo de 2026 a partir del análisis y corrección del notebook `tp1/tp_cdd.ipynb`.*
