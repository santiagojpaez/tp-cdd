# Reporte de análisis y evaluación del notebook `tp2_cdd.ipynb`

**Dataset:** `intent.csv` — Clasificación de intenciones (Intent Detection) para chatbots  
**Fecha:** Mayo 2026  
**Asignatura:** Ciencia de Datos 2026  
**Notebook fuente:** `tp2/tp2_cdd.ipynb`

---

## Índice

1. [Contexto y estructura general](#1-contexto-y-estructura-general)
2. [Fase 1 — Carga y revisión inicial del dataset](#2-fase-1--carga-y-revisión-inicial-del-dataset)
3. [Fase 2 — Análisis Exploratorio de Datos (EDA)](#3-fase-2--análisis-exploratorio-de-datos-eda)
   - [3.1 Distribución de categorías](#31-distribución-de-categorías)
   - [3.2 Análisis de longitud de textos](#32-análisis-de-longitud-de-textos)
   - [3.3 Palabras más frecuentes y nubes de palabras](#33-palabras-más-frecuentes-y-nubes-de-palabras)
4. [Fase 3 — Preprocesamiento para modelado](#4-fase-3--preprocesamiento-para-modelado)
   - [4.1 Limpieza de texto](#41-limpieza-de-texto)
   - [4.2 Codificación de la variable objetivo](#42-codificación-de-la-variable-objetivo)
   - [4.3 Representaciones vectoriales: BoW y TF-IDF](#43-representaciones-vectoriales-bow-y-tf-idf)
   - [4.4 Balanceo del conjunto de datos con SMOTE](#44-balanceo-del-conjunto-de-datos-con-smote)
   - [4.5 División entrenamiento/prueba](#45-división-entrenamientoprueba)
5. [Fase 4 — Modelado y evaluación](#5-fase-4--modelado-y-evaluación)
   - [5.1 Modelos entrenados y búsqueda de hiperparámetros](#51-modelos-entrenados-y-búsqueda-de-hiperparámetros)
   - [5.2 Métricas de evaluación](#52-métricas-de-evaluación)
   - [5.3 Resultados obtenidos](#53-resultados-obtenidos)
   - [5.4 Análisis comparativo del rendimiento](#54-análisis-comparativo-del-rendimiento)
6. [Fase 5 — Notas finales y fundamentación de decisiones](#6-fase-5--notas-finales-y-fundamentación-de-decisiones)
   - [6.1 ¿Por qué TF-IDF supera a BoW?](#61-por-qué-tf-idf-supera-a-bow)
   - [6.2 ¿Por qué Logistic Regression supera a los ensembles?](#62-por-qué-logistic-regression-supera-a-los-ensembles)
   - [6.3 Estrategia de balanceo con SMOTE](#63-estrategia-de-balanceo-con-smote)
7. [Conclusiones y recomendaciones](#7-conclusiones-y-recomendaciones)

---

## 1. Contexto y estructura general

El presente documento constituye un reporte técnico detallado del trabajo de análisis y modelado contenido en el notebook `tp2_cdd.ipynb`, desarrollado como parte del segundo trabajo práctico de Ciencia de Datos aplicada a Procesamiento de Lenguaje Natural (NLP). El objetivo central es construir y evaluar modelos de clasificación capaces de identificar la intención (*intent*) de un usuario a partir de un mensaje de texto en inglés, una tarea fundamental en sistemas de chatbots y asistentes conversacionales.

El dataset utilizado es una versión adaptada del conjunto de datos propuesto por Casanueva et al. (2020) en el paper *Efficient Intent Detection with Dual Sentence Encoders*. Contiene **13,083 mensajes** etiquetados en **77 categorías de intención** distintas, abarcando dominios como consultas bancarias, tarjetas de crédito, transferencias y servicios al cliente.

El notebook se articula en tres partes técnicas principales. La **Parte 1** comprende el Análisis Exploratorio de Datos (EDA), donde se examina la distribución de categorías, las características de los textos y los patrones léxicos. La **Parte 2** abarca el preprocesamiento completo: limpieza de texto, codificación de la variable objetivo, generación de representaciones vectoriales (Bag of Words y TF-IDF), balanceo de clases con SMOTE y división entrenamiento/prueba. La **Parte 3** implementa el entrenamiento de tres modelos de clasificación con búsqueda de hiperparámetros mediante GridSearchCV y validación cruzada estratificada de 3 folds, seguido de una evaluación comparativa exhaustiva.

El stack tecnológico incluye `pandas`, `numpy`, `matplotlib` y `seaborn` para análisis y visualización; `scikit-learn` para vectorización, modelado y evaluación; `imbalanced-learn` para balanceo con SMOTE; y `nltk` con `wordcloud` para el análisis léxico.

---

## 2. Fase 1 — Carga y revisión inicial del dataset

La primera fase establece las bases del análisis mediante una inspección estructural del dataset. Se carga el archivo `intent.csv` y se despliegan sus propiedades fundamentales.

El dataset contiene **13,083 filas y 2 columnas**:
- `text`: Mensaje del usuario (string), con textos en inglés de longitud variable.
- `category`: Intención detectada (string), con 77 valores únicos.

No se detectan valores nulos en ninguna columna. Los tipos de datos son correctos: ambas columnas son de tipo `object` (string). Esta simplicidad estructural —solo dos columnas— es característica de problemas de NLP donde toda la información predictiva reside en el texto crudo, y la tarea consiste en transformar esa secuencia de caracteres en una representación numérica que los algoritmos de machine learning puedan procesar.

---

## 3. Fase 2 — Análisis Exploratorio de Datos (EDA)

El EDA se despliega en tres movimientos analíticos complementarios: distribución de categorías, análisis de longitud de textos y exploración del contenido léxico mediante frecuencias y nubes de palabras.

### 3.1 Distribución de categorías

El análisis de la variable objetivo revela una distribución con desbalance moderado:

| Estadístico | Valor |
|-------------|-------|
| Total de categorías | 77 |
| Media de muestras por clase | 169.9 |
| Clase mayoritaria (`card_payment_fee_charged`) | 227 |
| Clase minoritaria (`contactless_not_working`) | 75 |
| Clases con <100 muestras | 3 |
| Clases con <150 muestras | 17 |

Las categorías más frecuentes pertenecen al dominio bancario/financiero: cobros de tarjeta, pagos no reconocidos, depósitos, retiros de efectivo. Las menos frecuentes incluyen problemas con tarjetas virtuales y funcionalidades contactless, reflejando posiblemente una menor frecuencia de consultas sobre estos temas en el contexto del dataset original.

Este desbalance, aunque no extremo (ratio ~3:1 entre la clase más y menos poblada), justifica la aplicación de técnicas de balanceo para evitar que los modelos sesguen sus predicciones hacia las clases mayoritarias.

### 3.2 Análisis de longitud de textos

| Métrica | Caracteres | Palabras |
|---------|------------|----------|
| Media | 58.2 | 11.7 |
| Mediana | 46.0 | 10.0 |
| Mínimo | 13.0 | 2.0 |
| Máximo | 433.0 | 79.0 |
| Desvío estándar | 39.6 | 7.6 |

Los textos son notablemente cortos, con una mediana de 46 caracteres y 10 palabras. Esto es esperable en interfaces conversacionales, donde los usuarios tienden a expresar sus intenciones de forma concisa. La distribución presenta asimetría positiva: la mayoría de los mensajes se concentra en el rango de 20–80 caracteres, con una cola derecha que se extiende hasta los 433 caracteres.

La longitud mínima de 2 palabras en algunos casos podría presentar desafíos para la clasificación, ya que contienen poca información léxica. Sin embargo, en el contexto de intent detection, frases como "lost card" o "balance update" contienen palabras clave altamente informativas que permiten una clasificación precisa incluso con pocas palabras.

### 3.3 Palabras más frecuentes y nubes de palabras

El análisis de frecuencia léxica global (post-limpieza: lowercase, sin puntuación, sin stopwords, tokens > 2 caracteres) revela **68,055 tokens** con un vocabulario de aproximadamente 8,000 términos únicos. Las palabras más frecuentes a nivel global incluyen términos específicos del dominio: *card*, *account*, *money*, *payment*, *transfer*, *bank*, *balance*, *time*, *cash*, *statement*, *like*, *one*, *get*, *need*, *want*.

Las nubes de palabras por categoría muestran patrones léxicos distintivos:
- **card_payment_fee_charged**: *fee*, *charged*, *card*, *payment*, *extra*, *money*
- **direct_debit_payment_not_recognised**: *payment*, *direct*, *debit*, *recognised*, *account*
- **balance_not_updated_after_cheque_or_cash_deposit**: *balance*, *deposit*, *cash*, *cheque*, *updated*
- **cash_withdrawal_charge**: *cash*, *withdrawal*, *atm*, *charged*, *fee*

Estos patrones confirman que cada categoría tiene un perfil léxico distintivo, lo cual es una señal positiva para la clasificación supervisada.

---

## 4. Fase 3 — Preprocesamiento para modelado

### 4.1 Limpieza de texto

La función `clean_text()` aplica transformaciones estándar de preprocesamiento de NLP:
1. **Lowercasing**: Conversión a minúsculas para unificar variantes.
2. **Eliminación de puntuación**: Remoción de signos que no aportan valor semántico.
3. **Tokenización y filtrado**: División en tokens, eliminación de stopwords y filtrado de tokens con longitud ≤2 caracteres.
4. **Recomposición**: Unión de tokens en un string limpio.

### 4.2 Codificación de la variable objetivo

Las 77 categorías de intención se codifican numéricamente mediante `LabelEncoder` de scikit-learn, asignando un entero único (0–76) a cada categoría.

### 4.3 Representaciones vectoriales

Se implementan tres representaciones: dos basadas en frecuencia (BoW y TF-IDF) y una basada en transformers (Sentence Embeddings).

#### 4.3.1 Bag of Words (CountVectorizer)

- Representa cada documento como un vector de frecuencias absolutas de términos.
- `max_features=3000`. Vocabulario real: **2,471 términos**.
- Shape resultante: (13083, 2471), matriz dispersa `csr_matrix`.

#### 4.3.2 TF-IDF (TfidfVectorizer)

- Pondera cada término por su frecuencia (TF) y su rareza en el corpus (IDF).
- Shape resultante: (13083, 2471), matriz dispersa `csr_matrix`.
- Términos con mayor IDF: *topup*, *dormant*, *virtual*, *beneficiary*, *cheque*, *expire*, *fiat*, *cryptocurrency*, *verification*, *refund*, *fee*, *pin*.
- La matriz dispersa mantiene el uso de memoria en ~2-3 MB vs ~266 MB si fuera densa.

#### 4.3.3 Sentence Embeddings (Sentence-BERT)

Como representación adicional moderna, se utiliza **Sentence-BERT** con el modelo `all-MiniLM-L6-v2`:

- Genera **embeddings densos de 384 dimensiones** (dense, float32).
- Captura semántica contextual: frases como "can't access my account" y "unable to log in" producen vectores cercanos aunque no compartan vocabulario.
- Limpieza más ligera que BoW/TF-IDF: solo lowercase y remoción de puntuación, preservando stopwords que aportan estructura sintáctica al transformer.
- Shape resultante: (13083, 384).
- **Nota:** Por ser valores continuos con negativos, se usa `GaussianNB` en lugar de `MultinomialNB` para Naive Bayes.

### 4.4 Balanceo del conjunto de datos con SMOTE

Dado el desbalance moderado entre las 77 clases, se aplica **SMOTE (Synthetic Minority Oversampling Technique)** con `k_neighbors=4`. Este parámetro se elige para garantizar compatibilidad con la clase minoritaria (75 muestras).

Resultado post-SMOTE (para cada representación):
- **Antes**: 13,083 muestras, distribución de 75 a 227 por clase.
- **Después**: 17,479 muestras, exactamente 227 por cada una de las 77 clases.
- Incremento del 33.6% en el tamaño del dataset.
- SMOTE se aplica por separado a BoW, TF-IDF y Embeddings, generando conjuntos balanceados independientes para cada representación.

### 4.5 División entrenamiento/prueba

El dataset balanceado se divide en entrenamiento (70%) y prueba (30%) mediante `train_test_split` con estratificación, garantizando que cada clase mantenga su proporción exacta en ambos conjuntos. Semilla aleatoria fijada en 42 para reproducibilidad.

- **Train**: 12,235 muestras
- **Test**: 5,244 muestras

---

## 5. Fase 4 — Modelado y evaluación

### 5.1 Modelos entrenados y búsqueda de hiperparámetros

Se entrenan y optimizan tres modelos con `GridSearchCV` utilizando validación cruzada estratificada de 3 folds y `f1_weighted` como métrica de optimización.

| # | Modelo | Hiperparámetros explorados | Combinaciones |
|---|--------|---------------------------|:---:|
| 1 | **Regresión Logística** | `C` ∈ {0.1, 1, 10}, `solver='lbfgs'` | 3 |
| 2 | **Naive Bayes** | MultinomialNB: `alpha` ∈ {0.01, 0.1, 0.5, 1.0} / GaussianNB: `var_smoothing` ∈ {1e-12, 1e-9, 1e-6} | 4 / 3 |
| 3 | **Random Forest** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {30, None}, `min_samples_split` ∈ {2, 5} | 8 |


La búsqueda se realiza sobre las tres representaciones (BoW, TF-IDF, Embeddings), resultando en 3 × (3+4+8) = 45 combinaciones de modelo × vectorización, cada una con 3-fold CV. Para embeddings se usa `GaussianNB` (en lugar de `MultinomialNB`) porque los vectores contienen valores negativos.

### 5.2 Métricas de evaluación

Dado que el problema tiene 77 clases balanceadas post-SMOTE, se utilizan:

- **Accuracy**: Proporción global de predicciones correctas.
- **Precision / Recall / F1 (weighted)**: Promedios ponderados por soporte de clase.
- **F1 (macro)**: Promedio simple por clase — penaliza mal desempeño en clases minoritarias.
- **Classification Report**: Desglose completo por clase.
- **Tiempo de entrenamiento**: Incluye GridSearchCV completo.

### 5.3 Resultados obtenidos

| Modelo | Vectorización | Accuracy | F1 (weighted) | F1 (macro) | Tiempo (s) |
|--------|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | BoW | 0.8604 | 0.8616 | 0.8615 | 11.4 |
| Multinomial NB | BoW | 0.8410 | 0.8407 | 0.8406 | **1.6** |
| Random Forest | BoW | 0.8448 | 0.8456 | 0.8456 | 26.5 |
| **Logistic Regression** | **TF-IDF** | **0.9068** | **0.9071** | **0.9070** | **5.7** |
| Multinomial NB | TF-IDF | 0.8873 | 0.8869 | 0.8868 | 2.3 |
| Random Forest | TF-IDF | 0.8955 | 0.8950 | 0.8949 | 35.5 |
| Logistic Regression | Embeddings | — | — | — | — |
| Gaussian NB | Embeddings | — | — | — | — |
| Random Forest | Embeddings | — | — | — | — |

*(Los resultados de embeddings se completan al ejecutar el notebook completo.)*

**Mejores hiperparámetros encontrados:**

| Modelo | Vectorización | Mejor configuración |
|--------|:---:|-------|
| Logistic Regression | TF-IDF | C = 10, solver = lbfgs |
| Multinomial NB | TF-IDF | alpha = 0.1 |
| Random Forest | TF-IDF | n_estimators = 200, max_depth = None, min_samples_split = 2 |

### 5.4 Análisis comparativo del rendimiento

El ranking final, ordenado por F1-score weighted, es:

1. **Logistic Regression + TF-IDF**: F1 = 0.9071, Accuracy = 0.9068, Tiempo = 5.7s. Claramente el mejor modelo en todas las métricas y con el segundo mejor tiempo de entrenamiento. La regularización L2 con C=10 controla eficazmente el sobreajuste en el espacio de 2,471 dimensiones.

2. **Random Forest + TF-IDF**: F1 = 0.8950, Accuracy = 0.8955, Tiempo = 35.5s. Segundo lugar en accuracy. Los árboles completamente desarrollados (max_depth=None) con 200 estimadores capturan bien las interacciones entre features, pero el costo computacional es 6× mayor que LR para una ganancia marginal negativa.

3. **Multinomial NB + TF-IDF**: F1 = 0.8869, Accuracy = 0.8873, Tiempo = 2.3s. El modelo más rápido y con rendimiento muy competitivo. Ideal como baseline rápido o para escenarios con restricciones severas de cómputo.

4. **Regresión Logística + BoW**: F1 = 0.8616, Tiempo = 11.4s. El drop de ~4.5 puntos de F1 respecto a TF-IDF muestra el valor de la ponderación por importancia relativa.

**Hallazgos clave:**

- **TF-IDF supera a BoW en todos los modelos**. La ganancia es particularmente notable en LR (+4.5 puntos de F1) y NB (+4.6 puntos).
- **Logistic Regression es el mejor modelo global**, contradiciendo la expectativa habitual de que los ensembles dominan. En un espacio de features dispersas de alta dimensionalidad con clases bien balanceadas, un modelo lineal bien regularizado puede ser óptimo.
- **Naive Bayes es el más rápido** (1.6-2.3s) con rendimiento competitivo, confirmando su utilidad como baseline y para iteración rápida.
- **La diferencia F1 weighted vs macro es mínima** en todos los modelos (<0.001), indicando que SMOTE logró un balanceo efectivo y que ningún modelo está sesgando sus predicciones hacia clases específicas.

**Reporte de clasificación del mejor modelo (LR + TF-IDF):**

El classification report completo muestra que la mayoría de las 77 clases tienen F1-scores entre 0.80 y 1.00. Las clases con mejor desempeño (~0.99 F1) incluyen `age_limit`, `apple_pay_or_google_pay`, `contactless_not_working`, `activate_my_card`, `atm_support`, `automatic_top_up`, `card_about_to_expire`, `change_pin`, `compromised_card`, `supported_cards_and_currencies`. Estas categorías tienen vocabulario muy distintivo y poco solapamiento con otras.

Las clases con peor desempeño (~0.79-0.84 F1) incluyen `balance_not_updated_after_bank_transfer` (F1=0.79), `card_arrival` (F1=0.84), `card_delivery_estimate` (F1=0.84). El solapamiento léxico entre estas categorías y otras relacionadas con "balance" y "card" explica la mayor dificultad de clasificación.

---

## 6. Fase 5 — Notas finales y fundamentación de decisiones

### 6.1 ¿Por qué TF-IDF supera a BoW?

TF-IDF pondera cada término por su importancia relativa: penaliza palabras frecuentes en todo el corpus y realza términos específicos de pocas categorías. En un problema con 77 clases donde muchas comparten vocabulario base (ej. "card", "account", "payment"), esta distinción es crítica.

El análisis de los términos con mayor IDF confirma esta hipótesis: palabras como *topup*, *cryptocurrency*, *beneficiary*, *cheque*, *dormant*, *virtual*, *fiat* son extremadamente específicas de ciertas categorías y reciben pesos altos en TF-IDF, mientras que palabras ubicuas como *card* reciben pesos bajos. Esto permite que incluso modelos lineales como Logistic Regression aprendan fronteras de decisión efectivas.

### 6.2 ¿Por qué Logistic Regression supera a los ensembles?

Este resultado —LR superando a Random Forest— puede parecer contraintuitivo, pero tiene explicaciones sólidas:

1. **Alta dimensionalidad + muestra moderada**: Con 2,471 features y 12,235 muestras de entrenamiento, el ratio features/muestras (~0.2) es manejable para un modelo lineal regularizado. Los ensembles basados en árboles, en cambio, tienden a sobreajustar en espacios de alta dimensionalidad si no se restringe agresivamente la profundidad.

2. **Features naturalmente lineales**: En BoW/TF-IDF, la presencia/ausencia de ciertas palabras es inherentemente aditiva para la clasificación de intenciones. Si la palabra "topup" aparece, la intención probablemente es `automatic_top_up`. Esta naturaleza casi lineal del problema favorece a LR.

3. **Regularización efectiva**: El valor óptimo C=10 (el más alto explorado, menor regularización) sugiere que el modelo se beneficia de mayor flexibilidad, pero la regularización L2 aún controla el sobreajuste mejor que los mecanismos de los árboles en este espacio.

4. **Balanceo perfecto post-SMOTE**: Con clases perfectamente balanceadas, LR no sufre el sesgo hacia clases mayoritarias que a veces favorece a los ensembles.

### 6.3 Estrategia de balanceo con SMOTE

Se eligió SMOTE con `k_neighbors=4` sobre otras alternativas porque:
- **RandomUnderSampler** habría eliminado ~4,000 muestras (~30% del dataset), perdiendo información valiosa.
- **RandomOverSampler** habría duplicado muestras sin agregar variabilidad.
- **class_weight='balanced'** es una alternativa válida pero no garantiza que el modelo vea suficientes ejemplos de clases minoritarias durante el entrenamiento.

La cercanía entre F1 weighted y F1 macro (diferencia <0.001 en todos los modelos) valida que SMOTE logró un balanceo efectivo.

### 6.4 Sentence Embeddings: ventajas esperadas y trade-offs

La inclusión de Sentence-BERT como tercera representación permite evaluar si la semántica contextual ofrece ventajas sobre enfoques puramente léxicos (BoW/TF-IDF):

**Ventajas esperadas:**
- **Semántica contextual**: "can't access my account" y "unable to log in" producen vectores cercanos aunque no compartan vocabulario. Esto debería mejorar la clasificación de frases con variaciones léxicas.
- **Dimensionalidad reducida**: 384 dimensiones vs 2,471 de BoW/TF-IDF, lo que acelera el entrenamiento de modelos complejos (especialmente RF) y reduce el riesgo de sobreajuste.
- **Robustez a out-of-vocabulary**: El tokenizador de BERT maneja palabras no vistas mediante subword tokenization.

**Desventajas:**
- **Costo de generación**: ~20-30 segundos para generar embeddings de 13K textos (one-time cost).
- **Dependencia externa**: El modelo `all-MiniLM-L6-v2` pesa ~80 MB.
- **Interpretabilidad reducida**: A diferencia de TF-IDF, no se puede inspeccionar directamente qué palabras impulsan cada predicción.
- **NB requiere adaptación**: Los embeddings tienen valores negativos, por lo que se debe usar `GaussianNB` en lugar de `MultinomialNB`.

---

## 7. Conclusiones y recomendaciones

### Sobre el dataset
- El dataset de 13,083 mensajes con 77 categorías presenta desbalance moderado (ratio 3:1), manejable con SMOTE.
- Los textos son cortos (mediana de 10 palabras), típicos de chatbots, haciendo que cada palabra tenga alto valor informativo.
- Cada categoría tiene un perfil léxico distintivo, confirmado por nubes de palabras y el alto rendimiento de los clasificadores.

### Sobre las representaciones vectoriales
- **TF-IDF supera consistentemente a BoW** para LR (+4.5pp F1), NB (+4.6pp) y RF (+4.9pp). La ponderación por importancia relativa es crítica con 77 clases que comparten vocabulario base.
- **Sentence Embeddings** (384 dims) ofrecen semántica contextual y menor dimensionalidad, potencialmente mejorando la generalización sobre variaciones léxicas que BoW/TF-IDF no capturan.
- Las matrices dispersas (`csr_matrix`) mantienen el uso de memoria en ~2-3 MB para BoW/TF-IDF. Embeddings ocupan ~27 MB (denso float32).
- El vocabulario de 2,471 términos demuestra que `max_features=3000` fue un límite superior adecuado.

### Sobre los modelos
- **Logistic Regression + TF-IDF es el claro ganador**: F1 0.9071, Accuracy 0.9068, entrenamiento en 5.7 segundos. Mejor relación accuracy/velocidad/interpretabilidad.
- **Multinomial NB + TF-IDF es el segundo mejor**: F1 0.8869 en solo 2.3 segundos. Ideal para iteración rápida o entornos con recursos limitados.
- **Random Forest + TF-IDF** rinde bien (F1 0.8950) pero es 6× más lento que LR sin ganancia de accuracy.

### Recomendación para producción
**Logistic Regression + TF-IDF** con C=10. Ofrece:
- **Accuracy**: 90.7% sobre 77 clases balanceadas.
- **Velocidad**: 5.7s de entrenamiento, inferencia instantánea.
- **Interpretabilidad**: Los coeficientes por clase indican exactamente qué palabras impulsan cada predicción.
- **Simplicidad operativa**: Un único modelo lineal, fácil de serializar y desplegar.

Si la velocidad de iteración es prioritaria (ej. experimentación frecuente), **Multinomial NB + TF-IDF** ofrece 88.7% F1 en 2.3s, siendo un excelente baseline rápido.

---

**Referencia**: Casanueva, I., Temcinas, T., Gerz, D., Henderson, M. y Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020.
