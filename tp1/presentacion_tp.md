# Trabajo Practico: Red Neuronal MLP para Mantenimiento Predictivo Industrial

## Integracion de Ciencia de Datos e Inteligencia Artificial

---

## 1. Introduccion y Objetivos

### 1.1 Contexto del Trabajo

Este Trabajo Practico integra los contenidos de **Ciencia de Datos (CD)** e **Inteligencia Artificial (IA)** mediante un pipeline unico de preprocesamiento y evaluacion. El notebook de CD (`tp_cdd.ipynb`) limpia, transforma, balancea, escala y exporta los datos. El notebook de IA (`tp_ia.ipynb`) consume esos archivos exportados para entrenar una red neuronal MLP y comparar sus resultados contra los modelos tradicionales.

La idea central es evitar dos pipelines distintos que produzcan resultados no comparables. CD define el dataset final; IA entrena sobre exactamente el mismo `train_data.csv` y evalua sobre exactamente el mismo `test_data.csv`.

### 1.2 Objetivos Especificos

1. Aplicar redes neuronales a un problema realista de clasificacion binaria.
2. Entrenar una arquitectura MLP con PyTorch sobre datos tabulares industriales.
3. Ajustar hiperparametros de manera sistematica y justificar la configuracion final.
4. Comparar el MLP contra modelos tradicionales entrenados en CD sobre el mismo split.
5. Documentar resultados consistentes con los archivos exportados y las salidas actuales de la notebook.

---

## 2. Dataset

### 2.1 Origen y Naturaleza de los Datos

El dataset `i40.csv` representa un escenario de **mantenimiento predictivo industrial**. Contiene mediciones operativas de maquinas y una variable objetivo que indica si la observacion corresponde a funcionamiento normal (`normal`) o a falla (`failure`).

### 2.2 Caracteristicas Iniciales

| Caracteristica | Valor |
|----------------|-------|
| Filas originales | 14,521 |
| Columnas originales | 9 |
| Variables predictoras utiles | 6 columnas originales: 5 numericas + `product_type` |
| Variables identificadoras descartadas | `idx`, `parent_device_id` |
| Variable objetivo | `target` (`normal` / `failure`) |
| Distribucion inicial | 51.53% `failure`, 48.47% `normal` |
| Valores nulos | 40 en `air_temp [K]` |
| Valores invalidos detectados | 47 registros con `speed [RPM] <= 0` |

### 2.3 Variables del Dataset

**Variables numericas:**

| Variable | Rango original | Observacion |
|----------|----------------|-------------|
| `air_temp [K]` | 295.3 - 304.5 | 40 nulos, imputados durante CD |
| `process_temp [K]` | 305.7 - 313.8 | Alta correlacion con `air_temp [K]` |
| `speed [RPM]` | -1 - 2886 | 47 valores fisicamente invalidos tratados como faltantes |
| `torque [Nm]` | 3.8 - 76.6 | Alta correlacion negativa con `speed [RPM]` |
| `tool_wear [min]` | 0 - 253 | Desgaste acumulado de herramienta |

**Variables categoricas:**

| Variable | Distribucion |
|----------|--------------|
| `product_type` | L: 71.36%, M: 18.53%, H: 10.11% |
| `target` | `failure`: 51.53%, `normal`: 48.47% |

---

## 3. Pipeline de Preprocesamiento de CD

### 3.1 Integracion CD-IA

El notebook `tp_cdd.ipynb` es la fuente de verdad del preprocesamiento. Exporta:

| Archivo | Uso |
|---------|-----|
| `train_data.csv` | Entrenamiento y validacion interna del MLP |
| `test_data.csv` | Evaluacion final del MLP |
| `resultados_modelos.csv` | Metricas finales de modelos tradicionales de CD |

El notebook `tp_ia.ipynb` carga esos archivos directamente. Esto elimina duplicacion de limpieza y asegura comparabilidad directa entre modelos.

### 3.2 Pasos Aplicados en CD

1. Eliminacion de identificadores: `idx` y `parent_device_id`.
2. Eliminacion de 72 filas duplicadas.
3. Correccion de `speed [RPM] <= 0`: esos valores se trataron como faltantes.
4. Imputacion bivariada con KMeans para pares correlacionados.
5. Conservacion de variables correlacionadas cuando aportan informacion de dominio.
6. Tratamiento de outliers con criterio IQR, sin clipping sobre `speed [RPM]` por representar regimenes operativos reales.
7. One-hot encoding de `product_type` con `drop_first=True`.
8. Binarizacion de `target`: `failure -> 1`, `normal -> 0`.
9. Balanceo con `RandomUnderSampler` hasta distribucion 50/50.
10. Escalado z-score con `StandardScaler` sobre columnas numericas.
11. Split estratificado 70/30 y exportacion de `train_data.csv` y `test_data.csv`.

### 3.3 Correccion Importante del Pipeline

El pipeline actual reemplaza `Normalizer()` por `StandardScaler()`.

Esto es clave porque `Normalizer()` escala por fila y proyecta cada muestra a norma unitaria, mientras que `StandardScaler()` estandariza por columna con media 0 y desvio estandar 1. Para datos tabulares, regresion logistica, KNN y redes neuronales, la estandarizacion por columna es la transformacion esperada.

### 3.4 Datos Exportados para IA

| Conjunto | Filas | Columnas totales | Features | Distribucion target |
|----------|------:|-----------------:|---------:|---------------------|
| Train | 9,854 | 8 | 7 | 4,927 normal / 4,927 failure |
| Test | 4,224 | 8 | 7 | 2,112 normal / 2,112 failure |

Columnas exportadas:

| Tipo | Columnas |
|------|----------|
| Numericas estandarizadas | `air_temp [K]`, `process_temp [K]`, `speed [RPM]`, `torque [Nm]`, `tool_wear [min]` |
| One-hot | `product_type_L`, `product_type_M` |
| Target | `target` |

La categoria `product_type_H` queda implicita cuando `product_type_L=False` y `product_type_M=False`.

---

## 4. Modelo MLP en IA

### 4.1 Arquitectura Base

El modelo implementado en `tp_ia.ipynb` es un MLP de clasificacion binaria:

```text
Input (7 features) -> [64] -> [32] -> [16] -> Output (1)
```

Cada capa oculta contiene:

| Componente | Funcion |
|------------|---------|
| Linear | Transformacion afin |
| BatchNorm1d | Estabilizacion de activaciones |
| ReLU | No linealidad |
| Dropout | Regularizacion |

La salida usa una neurona con `Sigmoid`, por lo que devuelve una probabilidad de falla entre 0 y 1.

### 4.2 Parametros del Modelo Base

| Elemento | Valor |
|----------|------:|
| Input features | 7 |
| Capas ocultas base | [64, 32, 16] |
| Dropout base de arquitectura | 0.4 |
| Parametros entrenables | 3,361 |
| Funcion de perdida | Binary Cross-Entropy |
| Optimizador usado en modelo final | Adam |

---

## 5. Entrenamiento y Early Stopping

### 5.1 Configuracion General

| Parametro | Valor |
|-----------|------:|
| Epocas maximas | 200 |
| Patience | 15 |
| Split interno train/validacion | 85% / 15% del train exportado |
| Tune set | 8,375 muestras |
| Validation set | 1,479 muestras |
| Test set final | 4,224 muestras, reservado hasta la evaluacion final |

### 5.2 Criterio de Seleccion

La metrica principal del tuning es **F1-score en validacion**. Esta eleccion es adecuada porque el problema de mantenimiento predictivo penaliza tanto falsos positivos como falsos negativos:

| Error | Interpretacion |
|-------|----------------|
| Falso positivo | Parada o inspeccion innecesaria |
| Falso negativo | Falla no detectada |

El AUC-ROC se usa como metrica secundaria para medir capacidad discriminativa independiente del umbral.

---

## 6. Ajuste de Hiperparametros

### 6.1 Espacio de Busqueda

Se exploraron **72 combinaciones**:

| Hiperparametro | Valores |
|----------------|---------|
| Arquitectura | [32,16], [64,32], [64,32,16] |
| Learning rate | 0.01, 0.001, 0.0005 |
| Dropout | 0.3, 0.5 |
| Batch size | 32, 64 |
| Optimizador | Adam, SGD con momentum=0.9 |

### 6.2 Top 5 del Tuning Actual

Valores tomados de la salida actual de `tp_ia.ipynb`.

| # | Arquitectura | lr | Dropout | Batch | Opt | Val F1 | Val AUC | Mejor epoca | Tiempo |
|---|--------------|---:|--------:|------:|-----|-------:|--------:|-------------:|-------:|
| 1 | [64,32,16] | 0.0100 | 0.3 | 64 | Adam | **0.9533** | 0.9881 | 34 | 37.6s |
| 2 | [64,32] | 0.0010 | 0.3 | 64 | Adam | 0.9524 | **0.9896** | 123 | 77.1s |
| 3 | [64,32,16] | 0.0100 | 0.3 | 32 | Adam | 0.9503 | 0.9867 | 36 | 68.9s |
| 4 | [64,32] | 0.0005 | 0.3 | 32 | Adam | 0.9500 | 0.9874 | 91 | 102.8s |
| 5 | [64,32] | 0.0005 | 0.3 | 64 | Adam | 0.9489 | 0.9881 | 132 | 97.3s |

### 6.3 Lectura del Tuning

| Hallazgo | Evidencia |
|----------|-----------|
| Adam domina el top 5 | Todas las mejores configuraciones usan Adam |
| Dropout 0.3 domina el top 5 | Todas las mejores configuraciones usan dropout 0.3 |
| Batch 64 es competitivo | Gana la mejor configuracion y aparece 3 veces en top 5 |
| lr=0.01 no fue malo en esta ejecucion | La mejor configuracion usa lr=0.01 |
| [64,32] queda muy cerca de [64,32,16] | F1 0.9524 vs 0.9533 |

### 6.4 Configuracion Ganadora

```text
Arquitectura:  [64, 32, 16]
Learning rate: 0.01
Dropout:       0.3
Batch size:    64
Optimizador:   Adam
Val F1:        0.9533
Val AUC:       0.9881
```

La configuracion ganadora en validacion no es la arquitectura base original con dropout 0.4, sino la mejor combinacion encontrada en el grid: dropout 0.3, Adam, batch 64 y learning rate 0.01.

---

## 7. Resultados del MLP

### 7.1 Reentrenamiento Final

Despues del tuning, el notebook reentrena el mejor modelo usando el train exportado y una validacion interna para early stopping. El test permanece reservado hasta el final.

| Resultado | Valor |
|-----------|------:|
| Mejor epoca final | 59 |
| Early stopping | Epoca 74 |
| Tiempo de entrenamiento final | 54.7s |

### 7.2 Metricas en Test

Valores tomados de la salida actual de `tp_ia.ipynb`.

| Metrica | Valor |
|---------|------:|
| Accuracy | 0.9399 |
| Precision | 0.9192 |
| Recall | 0.9645 |
| **F1-Score** | **0.9413** |
| **ROC AUC** | **0.9865** |

### 7.3 Matriz de Confusion del MLP

| Real \ Predicho | normal | failure |
|-----------------|-------:|--------:|
| normal | 1,933 | 179 |
| failure | 75 | 2,037 |

El MLP prioriza capturar fallas: recall de `failure` = 0.9645. Esto deja 75 falsos negativos sobre 2,112 fallas reales.

### 7.4 Curvas de Aprendizaje

Las curvas de loss y accuracy muestran convergencia con brecha acotada entre entrenamiento y validacion. Early stopping selecciona la epoca con mejor validacion antes de evaluar sobre test.

---

## 8. Comparacion con Modelos Tradicionales de CD

### 8.1 Resultados Exportados por CD

Valores tomados de `resultados_modelos.csv`.

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|--------|---------:|----------:|-------:|---------:|--------:|
| Regresion Logistica | 0.8378 | 0.8296 | 0.8504 | 0.8398 | 0.9219 |
| Naive Bayes | 0.7971 | 0.7577 | 0.8736 | 0.8115 | 0.8847 |
| KNN | 0.9512 | 0.9243 | **0.9830** | 0.9527 | 0.9754 |
| Arbol de Decision | 0.9524 | 0.9459 | 0.9598 | 0.9528 | 0.9524 |
| Random Forest | 0.9664 | 0.9522 | 0.9820 | 0.9669 | 0.9956 |
| **Gradient Boosting** | **0.9716** | **0.9624** | 0.9815 | **0.9719** | **0.9962** |
| **MLP (PyTorch)** | 0.9399 | 0.9192 | 0.9645 | 0.9413 | 0.9865 |

### 8.2 Ranking por F1-Score

| Puesto | Modelo | F1-Score |
|-------:|--------|---------:|
| 1 | Gradient Boosting | 0.9719 |
| 2 | Random Forest | 0.9669 |
| 3 | Arbol de Decision | 0.9528 |
| 4 | KNN | 0.9527 |
| 5 | MLP (PyTorch) | 0.9413 |
| 6 | Regresion Logistica | 0.8398 |
| 7 | Naive Bayes | 0.8115 |

### 8.3 Analisis Comparativo

| Comparacion | Lectura |
|-------------|---------|
| MLP vs modelos lineales/probabilisticos | MLP supera ampliamente a Regresion Logistica y Naive Bayes |
| MLP vs KNN/Arbol | MLP queda por debajo de KNN y Arbol de Decision en F1 |
| MLP vs ensembles | Random Forest y Gradient Boosting son superiores |
| AUC del MLP | 0.9865, muy competitivo aunque no lidera |
| Recall del MLP | 0.9645, bueno para detectar fallas |

El resultado es consistente con datos tabulares de tamano medio: los ensembles de arboles suelen dominar porque capturan interacciones no lineales y particiones locales sin requerir gran volumen de datos.

### 8.4 Comparacion de Tiempos

| Modelo | Tiempo reportado |
|--------|-----------------:|
| Naive Bayes | 0.06s |
| KNN | 0.7s |
| Arbol de Decision | 1.8s |
| Regresion Logistica | 4.6s |
| Random Forest | 30.6s |
| Gradient Boosting | 34.7s |
| MLP (PyTorch) | 54.7s |

El MLP fue mas costoso que los modelos tradicionales reportados. Su ventaja potencial aparece en escenarios con mas datos, GPU o arquitecturas neuronales especializadas.

---

## 9. Conclusiones

### 9.1 Conclusiones Tecnicas

1. El pipeline CD-IA quedo integrado: IA usa `train_data.csv`, `test_data.csv` y `resultados_modelos.csv` exportados por CD.
2. El cambio de `Normalizer()` a `StandardScaler()` modifica el benchmark y debe considerarse parte central de la version actual.
3. El MLP alcanza buen rendimiento absoluto: F1 0.9413 y AUC 0.9865.
4. El MLP no supera a los mejores modelos tradicionales en este dataset: Gradient Boosting lidera con F1 0.9719 y AUC 0.9962.
5. La mejor configuracion del MLP en la notebook actual usa [64,32,16], learning rate 0.01, dropout 0.3, batch 64 y Adam.
6. La arquitectura [64,32] queda muy cerca en validacion, por lo que seria una alternativa pragmatica si se prioriza menor complejidad.

### 9.2 Conclusiones sobre Integracion CD-IA

1. La comparacion es valida porque todos los modelos usan el mismo split exportado.
2. El test se mantiene como evaluacion final en IA, sin usarlo durante el tuning del MLP.
3. Los resultados hardcodeados de la presentacion deben tomar como fuentes de verdad `tp_ia.ipynb` para MLP y `resultados_modelos.csv` para modelos CD.

### 9.3 Lineas Futuras

1. Ajustar `weight_decay` para combinar L2 con dropout.
2. Probar scheduler de learning rate, especialmente si se quiere hacer competitivo SGD.
3. Evaluar modelos neuronales tabulares como TabNet o TabTransformer.
4. Analizar explicabilidad con SHAP o LIME.
5. Validar performance por `product_type` para detectar sesgos o dependencia excesiva de la categoria.
6. Si se dispone de historial temporal, explorar LSTM/GRU o CNN 1D para secuencias de sensores.

---

## 10. Referencias Bibliograficas

- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*.
- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR, 15*(1), 1929-1958.
- Bengio, Y. (2012). Practical Recommendations for Gradient-Based Training of Deep Architectures. *Neural Networks: Tricks of the Trade*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Masters, D., & Luschi, C. (2018). Revisiting Small Batch Training for Deep Neural Networks. *arXiv:1804.07612*.

---

## Anexo: Fuentes Usadas

| Recurso | Rol |
|---------|-----|
| `i40.csv` | Dataset original |
| `tp_cdd.ipynb` | Pipeline de preprocesamiento y modelos tradicionales |
| `train_data.csv` | Train exportado para IA |
| `test_data.csv` | Test exportado para IA |
| `resultados_modelos.csv` | Metricas exportadas de CD |
| `tp_ia.ipynb` | Entrenamiento, tuning y evaluacion del MLP |
| `tp_cdd_reporte.md` | Contexto tecnico de correcciones del pipeline |
