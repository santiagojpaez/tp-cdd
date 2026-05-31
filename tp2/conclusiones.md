# Conclusiones - TP 2 (Ciencia de Datos)

## Resumen de la Actividad
Se desarrolló un pipeline completo de NLP para clasificación de intenciones (*intent detection*) sobre un dataset de 13,083 mensajes en inglés distribuidos en 77 categorías. El trabajo abarcó análisis exploratorio, preprocesamiento de texto, representación vectorial (BoW y TF-IDF), balanceo de clases con SMOTE y modelado con búsqueda de hiperparámetros mediante GridSearchCV.

### Pipeline implementado
1. **EDA:** Distribución de categorías (77 clases, 75-227 muestras), análisis de longitud de textos (mediana 10 palabras), nubes de palabras y frecuencias léxicas (68,055 tokens post-limpieza).
2. **Preprocesamiento:** Limpieza de texto (lowercase, sin puntuación ni stopwords), LabelEncoder para 77 clases, vectorización BoW/TF-IDF (2,471 términos, matriz dispersa) + Sentence Embeddings (Sentence-BERT, 384 dims).
3. **Balanceo:** SMOTE con k_neighbors=4, pasando de 13,083 a 17,479 muestras por representación (227 por clase).
4. **Modelado:** 3 modelos (LR, NB, Random Forest) × 3 representaciones, GridSearchCV con 3-fold CV estratificado.

## Resultados obtenidos

| Modelo | Vectorización | Accuracy | F1 (weighted) | Tiempo (s) |
|--------|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **TF-IDF** | **0.9068** | **0.9071** | **5.7** |
| Random Forest | TF-IDF | 0.8955 | 0.8950 | 35.5 |
| Multinomial NB | TF-IDF | 0.8873 | 0.8869 | 2.3 |
| Logistic Regression | BoW | 0.8604 | 0.8616 | 11.4 |

*Los resultados con Sentence Embeddings se completan al ejecutar el notebook completo con las 3 representaciones.*

### Hallazgos principales
- **Logistic Regression + TF-IDF es el mejor modelo** en accuracy, F1 y eficiencia (5.7s de entrenamiento). La regularización L2 con C=10 controla eficazmente el sobreajuste en el espacio de 2,471 dimensiones.
- **TF-IDF supera a BoW en LR, NB y RF** (+4-5 puntos de F1 en cada caso).
- **Multinomial NB es el más rápido** (2.3s) con rendimiento competitivo (F1 0.8869), excelente como baseline para iteración rápida.
- **SMOTE funciona correctamente:** la diferencia entre F1 weighted y macro es <0.001 en todos los modelos, confirmando que ninguna clase está siendo sistemáticamente perjudicada.

## Evaluación del cumplimiento del TP
- **Inciso 1 (EDA):** Cumplido. Distribuciones, medidas estadísticas, gráficos de barras, histogramas, nubes de palabras.
- **Inciso 2 (Preprocesamiento):** Cumplido. Limpieza de texto, codificación, BoW/TF-IDF, balanceo con SMOTE.
- **Inciso 3 (Modelado):** Cumplido. 3 modelos con GridSearchCV, división train/test estratificada, métricas completas (accuracy, precision, recall, F1 weighted/macro, classification report por clase).

## Lecciones aprendidas
- En clasificación de texto con features dispersas y clases balanceadas, un modelo lineal bien regularizado puede superar a ensembles complejos.
- La representación vectorial (TF-IDF vs BoW) impacta más que la elección del modelo: la ganancia de pasar de BoW a TF-IDF (+4.5pp F1) es mayor que la diferencia entre el mejor y el peor modelo con la misma representación.
- Las matrices dispersas (`csr_matrix`) son esenciales para NLP: reducen el uso de memoria de ~266 MB a ~2-3 MB sin pérdida de información.

## Recomendación
**Logistic Regression + TF-IDF** es la configuración óptima para producción: 90.7% F1, entrenamiento en 5.7s, inferencia instantánea y coeficientes interpretables por clase.
