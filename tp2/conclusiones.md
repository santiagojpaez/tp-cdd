# Conclusiones - TP 2 (Ciencia de Datos)

## Resumen de la Actividad
Se desarrolló un pipeline completo de NLP para clasificación de intenciones (*intent detection*) sobre un dataset de 13,083 mensajes en inglés distribuidos en 77 categorías. El trabajo abarcó análisis exploratorio, preprocesamiento de texto, representación vectorial (BoW, TF-IDF y Sentence Embeddings), balanceo de clases con SMOTE y modelado con búsqueda de hiperparámetros mediante GridSearchCV.

### Pipeline implementado
1. **EDA:** Distribución de categorías (77 clases, 75-227 muestras), análisis de longitud de textos (mediana 10 palabras), nubes de palabras y frecuencias léxicas (68,055 tokens post-limpieza).
2. **Preprocesamiento:** Limpieza de texto (lowercase, sin puntuación ni stopwords), LabelEncoder para 77 clases, vectorización BoW/TF-IDF (2,471 términos, matriz dispersa) + Sentence Embeddings (Sentence-BERT, 384 dims).
3. **Balanceo:** SMOTE con k_neighbors=4, pasando de 13,083 a 17,479 muestras por representación (227 por clase).
4. **Modelado:** 3 modelos (LR, Naive Bayes, Random Forest) × 3 representaciones, GridSearchCV con 3-fold CV estratificado.

## Resultados obtenidos

| Modelo | Vectorización | Accuracy | F1 (weighted) | Tiempo (s) |
|--------|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **Embeddings** | **0.9405** | **0.9423** | **1.7** |
| Logistic Regression | TF-IDF | 0.9068 | 0.9071 | 5.7 |
| Random Forest | Embeddings | 0.9189 | 0.9198 | 12.3 |
| Random Forest | TF-IDF | 0.8955 | 0.8950 | 35.5 |
| Naive Bayes | TF-IDF | 0.8873 | 0.8869 | 2.3 |
| Logistic Regression | BoW | 0.8604 | 0.8616 | 11.4 |

### Hallazgos principales
- **Logistic Regression + Embeddings es el mejor modelo**: F1=0.9423, entrenamiento en 1.7s. Sentence-BERT captura semántica contextual que BoW/TF-IDF no pueden representar, y la dimensionalidad reducida (384 vs 2,471) acelera el entrenamiento.
- **Embeddings supera a TF-IDF por +3.5 puntos de F1** en Logistic Regression, confirmando que la semántica contextual es superior a enfoques puramente léxicos para intent detection.
- **TF-IDF supera a BoW en LR, NB y RF** (+4-5 puntos de F1 en cada caso). La ponderación por importancia relativa es crítica con 77 clases.
- **Naive Bayes es el más rápido en TF-IDF** (2.3s) con rendimiento competitivo (F1 0.8869), excelente como baseline para iteración rápida.
- **SMOTE funciona correctamente:** la diferencia entre F1 weighted y macro es <0.001 en todos los modelos, confirmando que ninguna clase está siendo sistemáticamente perjudicada.

## Evaluación del cumplimiento del TP
- **Inciso 1 (EDA):** Cumplido. Distribuciones, medidas estadísticas, gráficos de barras, histogramas, nubes de palabras.
- **Inciso 2 (Preprocesamiento):** Cumplido. Limpieza de texto, codificación, BoW/TF-IDF/Embeddings, balanceo con SMOTE.
- **Inciso 3 (Modelado):** Cumplido. 3 modelos con GridSearchCV, división train/test estratificada, métricas completas (accuracy, precision, recall, F1 weighted/macro, classification report por clase).

## Lecciones aprendidas
- En clasificación de texto con Sentence Embeddings, un modelo lineal simple (Logistic Regression) logra resultados excepcionales (F1=0.9423) gracias a la calidad de la representación semántica.
- La representación vectorial impacta más que la elección del modelo: Embeddings (+3.5pp sobre TF-IDF) y TF-IDF (+4.5pp sobre BoW) generan ganancias mayores que cambiar de algoritmo.
- Las matrices dispersas (`csr_matrix`) son esenciales para NLP: reducen el uso de memoria de ~266 MB a ~2-3 MB sin pérdida de información.
- No todos los problemas requieren modelos complejos: Logistic Regression bien regularizado sobre buenos embeddings supera a Random Forest por un margen considerable.

## Recomendación
**Logistic Regression + Embeddings (Sentence-BERT)** es la configuración óptima para producción: 94.2% F1, entrenamiento en 1.7s, inferencia instantánea. Como alternativa con total interpretabilidad de coeficientes, **Logistic Regression + TF-IDF** ofrece 90.7% F1 en 5.7s.
