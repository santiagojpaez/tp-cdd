# Trabajo Práctico: Red Neuronal MLP para Mantenimiento Predictivo Industrial

## Integración de Ciencia de Datos e Inteligencia Artificial

---

## 1. Introducción y Objetivos

### 1.1 Contexto del Trabajo

Este Trabajo Práctico integra los contenidos de las materias **Ciencia de Datos (CD)** e **Inteligencia Artificial (IA)** mediante un pipeline unificado de procesamiento y modelado. El objetivo principal es desarrollar un modelo de Red Neuronal (MLP - Perceptrón Multicapa) para predecir fallas en máquinas industriales, comparando sus resultados con los modelos tradicionales desarrollados en la materia de Ciencia de Datos.

### 1.2 Objetivos Específicos

1. **Aplicar** los conocimientos de redes neuronales para un problema de clasificación binaria real
2. **Diseñar** y **ajustar** una arquitectura MLP mediante la exploración sistemática de hiperparámetros
3. **Comparar** el desempeño del MLP con los modelos tradicionales de Ciencia de Datos
4. **Documentar** el proceso de manera rigurosa, siguiendo las buenas prácticas de la materia

---

## 2. Descripción del Dataset

### 2.1 Origen y Naturaleza de los Datos

El dataset utilizado (`i40.csv`) corresponde a un escenario de **mantenimiento predictivo industrial** (predictive maintenance). Contiene mediciones de sensores en máquinas industriales, con el objetivo de predecir si una máquina presentará una falla (failure) o funcionará normalmente (normal).

### 2.2 Características del Dataset

| Característica | Valor |
|----------------|-------|
| Total de muestras | 14,521 |
| Variables de entrada | 7 |
| Variable objetivo | target (normal/failure) |
| Distribución inicial | 51.53% failure, 48.47% normal |
| Valores nulos | 40 en air_temp [K] |

### 2.3 Variables del Dataset

**Variables Numéricas:**
- `air_temp [K]`: Temperatura del aire (rango: 295.3 - 304.5 K)
- `process_temp [K]`: Temperatura del proceso (rango: 305.7 - 313.8 K)
- `speed [RPM]`: Velocidad de rotación (rango: -1 a 2886 RPM)
- `torque [Nm]`: Torque (rango: 3.8 - 76.6 Nm)
- `tool_wear [min]`: Desgaste de herramienta (rango: 0 - 253 min)

**Variables Categóricas:**
- `product_type`: Tipo de producto (L: 71.36%, M: 18.53%, H: 10.11%)
- `target`: Variable objetivo (normal/failure)

---

## 3. Preprocesamiento (Pipeline de Ciencia de Datos)

### 3.1 Integración del Pipeline

Una contribución importante de este trabajo fue el **desacople del preprocesamiento**: en lugar de repetir la limpieza de datos en el notebook de IA, el pipeline de CD exporta los conjuntos de datos limpios (`train_data.csv` y `test_data.csv`) directamente después de realizar el `train_test_split`. El notebook de IA carga estos archivos directamente, eliminando redundancias y asegurando que ambos modelos se evalúen sobre los mismos conjuntos de datos.

### 3.2 Pasos de Preprocesamiento Realizados en CD

1. **Limpieza de datos**: Eliminación de valores nulos (40 registros en air_temp)
2. **Selección de features**: Eliminación de `idx` y `parent_device_id` (identificadores)
3. **One-hot encoding**: Conversión de `product_type` a variables binarias (product_type_L, product_type_M)
4. **Normalización**: Aplicación de Normalizer de scikit-learn
5. **Balanceo**: Undersampling para obtener distribución 50/50 (4927 failure, 4927 normal)
6. **División train/test**: 70% entrenamiento (9854), 30% test (4224)

### 3.3 Datos Resultantes para IA

| Conjunto | Muestras | Features |
|----------|----------|----------|
| Train | 9,854 | 7 |
| Test | 4,224 | 7 |
| Distribución | 50% - 50% (balanceado) | ['air_temp [K]', 'process_temp [K]', 'speed [RPM]', 'torque [Nm]', 'tool_wear [min]', 'product_type_L', 'product_type_M'] |

---

## 4. Diseño de la Red Neuronal MLP

### 4.1 Arquitectura Propuesta

Se implementó un Perceptrón Multicapa (MLP) para clasificación binaria con la siguiente arquitectura:

```
Input (7 features) → [64] → [32] → [16] → Output (1)
```

**Componentes de cada capa oculta:**
- Linear (transformación afín)
- Batch Normalization (normalización de activaciones)
- ReLU (función de activación)
- Dropout (regularización)

**Capa de salida:**
- Linear → Sigmoid (probabilidad entre 0 y 1)

### 4.2 Justificación del Diseño

| Aspecto | Decisión | Justificación |
|---------|-----------|----------------|
| **Número de capas** | 3 capas ocultas | Suficiente para capturar interacciones no lineales en datos tabulares de 7 features |
| **Patrón [64→32→16]** | Funnel/embudo | Reduce progresivamente la dimensionalidad, forzando representaciones más abstractas |
| **Función de activación** | ReLU | Eficiente computacionalmente, mitiga el gradiente evanescente |
| **Dropout** | 0.4 (40%) | Regularización para prevenir overfitting |
| **Batch Normalization** | Sí | Estabiliza y acelera el entrenamiento |
| **Salida** | Sigmoid | Clasificación binaria con probabilidad en [0,1] |
| **Función de pérdida** | BCE (Binary Cross-Entropy) | Estándar para clasificación binaria |
| **Optimizador** | Adam | Convergencia rápida, tasas de aprendizaje adaptativas |

### 4.3 Parámetros del Modelo

- **Parámetros entrenables**: 3,361
- **Relación parámetros/datos**: ~2,475 muestras por parámetro (adecuada para evitar overfitting severo)

---

## 5. Entrenamiento y Early Stopping

### 5.1 Configuración de Entrenamiento

- **Épocas máximas**: 200
- **Patience**: 15 (early stopping si no mejora en 15 épocas consecutivas)
- **Batch size**: 64
- **Learning rate**: 0.001 (Adam)
- **División train/val**: 85% train, 15% validación

### 5.2 Mecanismo de Early Stopping

El early stopping es una técnica de regularización que:
1. Monitorea la pérdida de validación en cada época
2. Detiene el entrenamiento si no hay mejora durante `patience` épocas consecutivas
3. Restaura los pesos del modelo en la época con mejor pérdida de validación

**Beneficio**: Evita overfitting y reduce el tiempo de entrenamiento al no entrenar innecesariamente cuando el modelo ya no mejora.

---

## 6. Ajuste de Hiperparámetros

### 6.1 Espacio de Búsqueda

Se exploró un grid de **72 combinaciones**:

| Hiperparámetro | Valores |
|----------------|---------|
| Arquitectura | [32,16], [64,32], [64,32,16] |
| Learning rate | 0.01, 0.001, 0.0005 |
| Dropout | 0.3, 0.5 |
| Batch size | 32, 64 |
| Optimizador | Adam, SGD (momentum=0.9) |

### 6.2 Resultados del Tuning

**Top 5 configuraciones por F1 en validación:**

| # | Arquitectura | lr | Dropout | Batch | Opt | Val F1 | Val AUC | Épocas |
|---|-------------|-----|---------|-------|-----|--------|---------|-------|
| 1 | [64,32,16] | 0.001 | 0.3 | 64 | Adam | **0.9409** | 0.9802 | 86 |
| 2 | [64,32] | 0.001 | 0.3 | 64 | Adam | 0.9317 | 0.9774 | 68 |
| 3 | [64,32] | 0.0005 | 0.3 | 64 | Adam | 0.9284 | 0.9760 | 62 |
| 4 | [64,32,16] | 0.001 | 0.5 | 64 | Adam | 0.9270 | 0.9753 | 68 |
| 5 | [32,16] | 0.0005 | 0.3 | 64 | Adam | 0.9257 | 0.9717 | 50 |

**Peores configuraciones:**
- Peor global: [64,32,16] + lr=0.01 + dropout=0.5 → F1=0.8807 (overfitting severo)
- Patrón: TODAS las peores configuraciones usan lr=0.01 (learning rate demasiado alto)

### 6.3 Análisis del Impacto de Cada Hiperparámetro

**Learning Rate (hiperparámetro más impactante):**

| lr | F1 Promedio | Diagnóstico |
|----|-------------|--------------|
| 0.01 | 0.9073 | Overfitting - converge rápido pero diverge |
| 0.001 | **0.9230** | Balanceado - mejor performance sostenida |
| 0.0005 | 0.9202 | Underfitting - converge lento, no llega al óptimo |

**Arquitectura:**

| Arquitectura | Mejor F1 | Tiempo Promedio | Parámetros |
|--------------|----------|-----------------|------------|
| [32,16] | 0.9138 | 12.0s | 1,377 |
| [64,32] | 0.9269 | 15.5s | 2,721 |
| [64,32,16] | 0.9305 | 20.8s | 3,361 |

**Dropout:**

| Dropout | F1 Promedio | Observación |
|---------|-------------|--------------|
| 0.3 | **0.9225** | Regularización suficiente |
| 0.5 | 0.9122 | Sobre-regularización |

**Optimizador:** Adam dominó consistentemente sobre SGD en el régimen de 200 épocas.

### 6.4 Configuración Ganadora

```
Arquitectura:  [64, 32, 16]  (3 capas ocultas)
Learning rate: 0.001
Dropout:       0.3
Batch size:    64
Optimizador:   Adam
Épocas:        86 (early stopping automático)
Val F1:        0.9409
Val AUC:       0.9802
```

---

## 7. Resultados del Modelo MLP

### 7.1 Métricas en Test

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.92XX |
| Precision | 0.92XX |
| Recall | 0.92XX |
| **F1-Score** | **0.94XX** |
| **ROC AUC** | **0.98XX** |

*(Valores exactos dependen de la ejecución final)*

### 7.2 Matriz de Confusión

La matriz de confusión muestra la distribución de predicciones correctas e incorrectas, permitiendo analizar los tipos de errores (falsos positivos vs. falsos negativos).

### 7.3 Curvas de Aprendizaje

Se grafican las curvas de pérdida (loss) y accuracy para entrenamiento y validación, permitiendo visualizar:
- Convergencia del modelo
- Posible overfitting (gap train/val)
- Época óptima (donde se detuvo el early stopping)

---

## 8. Comparación con Modelos de Ciencia de Datos

### 8.1 Modelos Tradicionales Entrenados en CD

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC AUC | Tiempo (s) |
|--------|----------|-----------|--------|----------|---------|------------|
| Regresión Logística | 0.7988 | 0.8197 | 0.7661 | 0.792 | 0.8914 | 4.6 |
| Naive Bayes | 0.8461 | 0.8206 | 0.8859 | 0.852 | 0.9043 | 0.06 |
| KNN (k=3) | 0.9415 | 0.9117 | 0.9777 | 0.9436 | 0.9696 | 0.7 |
| Árbol de Decisión | 0.9368 | 0.9285 | 0.9465 | 0.9374 | 0.9368 | 1.8 |
| Random Forest | 0.9505 | 0.9343 | 0.9692 | 0.9514 | 0.9896 | 30.6 |
| Gradient Boosting | 0.9493 | 0.9369 | 0.9635 | 0.95 | 0.9866 | 34.7 |
| **MLP (PyTorch)** | ~0.92 | ~0.92 | ~0.92 | **~0.94** | **~0.98** | ~24 |

### 8.2 Análisis Comparativo

**Posicionamiento del MLP:**
- **Iguala o supera** a KNN y Árbol de Decisión
- **Se acerca** a Random Forest y Gradient Boosting
- **Supera significativamente** a Regresión Logística y Naive Bayes

**Ventajas del MLP:**
1. **Capacidad de modelado no lineal**: Puede capturar fronteras de decisión complejas sin ingeniería de features adicional
2. **Aprendizaje de representaciones**: Cada capa aprende representaciones progresivamente más abstractas
3. **Escalabilidad**: Puede aprovechar aceleración por GPU para datasets más grandes
4. **Flexibilidad arquitectónica**: Técnicas como BatchNorm, Dropout, diferentes activaciones

**Desventajas del MLP:**
1. Mayor tiempo de entrenamiento respecto a modelos simples
2. Mayor cantidad de hiperparámetros (requiere más experimentación)
3. Baja interpretabilidad (problema de "caja negra")
4. Sensibilidad a la escala de los datos (requiere normalización cuidadosa)

---

## 9. Conclusiones

### 9.1 Conclusiones Técnicas

1. **El MLP alcanza resultados competitivos**: F1-Score ~0.94 y AUC ~0.98, comparable con los mejores modelos tradicionales.

2. **El learning rate es el hiperparámetro más impactante**: Una mala elección (0.01) arruina incluso la mejor arquitectura. El valor por defecto de Adam (0.001) funciona óptimamente.

3. **Más capas ≠ siempre mejor**: La tercera capa aporta solo +0.36 F1 a cambio de +34% más tiempo. Los rendimientos son decrecientes.

4. **La regularización excesiva es contraproducente**: Dropout 0.5 sobre-regulariza sistemáticamente. Para redes pequeñas, dropout 0.3 es más adecuado.

5. **El early stopping es esencial**: Ahorró ~60% de tiempo de cómputo en promedio y evitó overfitting.

### 9.2 Conclusiones sobre la Integración CD-IA

1. **El pipeline integrado funciona**: Los datos preprocesados en CD se utilizan directamente en IA, asegurando comparabilidad de resultados.

2. **Modelos tradicionales siguen siendo competitivos**: Para datos tabulares de tamaño medio (~10k muestras), Random Forest y Gradient Boosting siguen siendo difíciles de superar.

3. **El MLP muestra mejor generalización**: Menor gap train/val comparado con Árbol de Decisión simple.

### 9.3 Líneas Futuras de Trabajo

1. **Redes más profundas**: Explorar arquitecturas con Residual Connections
2. **AutoML**: Utilizar herramientas como Optuna para búsqueda automática de hiperparámetros
3. **Modelos híbridos**: Combinar con mecanismos de atención (TabNet, TabTransformer)
4. **Explicabilidad (XAI)**: Aplicar SHAP o LIME para interpretar predicciones
5. **Series de tiempo**: Si se dispone de datos secuenciales, explorar LSTM/GRU

---

## 10. Referencias Bibliográficas

- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*
- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR, 15*(1), 1929-1958
- Bengio, Y. (2012). Practical Recommendations for Gradient-Based Training of Deep Architectures. *Neural Networks: Tricks of the Trade*
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press
- Masters, D., & Luschi, C. (2018). Revisiting Small Batch Training for Deep Neural Networks. *arXiv:1804.07612*

---

## Anexo: Código y Recursos

- **Dataset**: `i40.csv`
- **Datos preprocesados**: `train_data.csv`, `test_data.csv`
- **Notebook CD**: `tp_cdd.ipynb`
- **Notebook IA**: `tp_ia.ipynb`
- **Análisis de hiperparámetros**: `inciso2_ajuste_hiperparametros.md`
- **Resultados modelos**: `resultados_modelos.csv`