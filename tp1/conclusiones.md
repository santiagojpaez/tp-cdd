# Conclusiones - Actividad 1 (Ciencia de Datos e Inteligencia Artificial)

## Resumen de la Actividad
Se realizó la integración del pipeline de preprocesamiento de la materia **Ciencia de Datos (CD)** con el modelado de redes neuronales de la materia **Inteligencia Artificial (IA)**. 

### Cambios principales realizados:
1. **Desacople del preprocesamiento:** En lugar de que el notebook de IA `tp_ia.ipynb` repita toda la limpieza y preprocesamiento de los datos, ahora el notebook `tp_cdd.ipynb` exporta los conjuntos de datos limpios (`train_data.csv` y `test_data.csv`) justo después de realizar el `train_test_split`.
2. **Carga directa en IA:** El notebook `tp_ia.ipynb` fue modificado para cargar directamente estos archivos CSV, eliminando las redundancias y asegurando que ambos modelos (los tradicionales de CD y la red neuronal de IA) se evalúen exactamente sobre los mismos conjuntos de entrenamiento y prueba.

## Análisis de la Arquitectura de la Red Neuronal (MLP)
La red implementada en `tp_ia.ipynb` es un Perceptrón Multicapa (MLP) diseñado para clasificación binaria (mantenimiento predictivo).

### Características del diseño:
- **Arquitectura tipo embudo (64 -> 32 -> 16 neuronas):** Se emplean 3 capas ocultas que reducen progresivamente la dimensionalidad. Esto permite a la red aprender representaciones cada vez más abstractas y complejas a partir de las 7 variables de entrada, sin sobrecargar la capacidad del modelo.
- **Función de activación ReLU:** Se utiliza en las capas ocultas para evitar el problema del gradiente evanescente y acelerar la convergencia.
- **Regularización (Dropout 40%):** Apaga aleatoriamente el 40% de las neuronas en cada paso de entrenamiento, siendo una medida robusta para evitar el sobreajuste (*overfitting*).
- **Batch Normalization:** Estabiliza la salida de cada capa antes de la activación, lo que acelera el entrenamiento y hace que la red sea menos sensible a la inicialización de los pesos.
- **Capa de salida:** Una única neurona con activación **Sigmoide** para devolver la probabilidad de fallo (valores entre 0 y 1).
- **Entrenamiento robusto:** Emplea el optimizador **Adam**, función de pérdida **Binary Cross-Entropy (BCE)**, y un mecanismo de **Early Stopping** para detener el entrenamiento si el *loss* de validación deja de mejorar, restaurando los mejores pesos encontrados.

## Evaluación del cumplimiento del TP
- **Integración de las materias:** Se cumple de forma estricta. El pipeline de CD genera los datos que consume la IA.
- **Comparabilidad:** Al usar exactamente el mismo `train_test_split` (mismos archivos CSV), las métricas obtenidas por la red neuronal son 100% comparables con las de Regresión Logística, Random Forest, etc.
- **Arquitectura:** El diseño del MLP es justificado, moderno y sigue buenas prácticas (Batch Normalization, Dropout, Early Stopping).

## Sugerencias de Mejora
1. **Optimización de hiperparámetros de la Red Neuronal:** Aunque el modelo base tiene una arquitectura sensata, se podrían emplear bibliotecas como *Optuna* para buscar la combinación ideal de tasa de aprendizaje, tamaño de *batch* y número de neuronas.
2. **Análisis de Importancia de Variables en IA:** Implementar técnicas como *SHAP* (SHapley Additive exPlanations) o *Integrated Gradients* para entender qué variables (ej. temperatura, torque, desgaste) están influyendo más en la decisión del MLP, aportando interpretabilidad al modelo "caja negra".
3. **Métricas de Negocio:** Más allá del F1-Score o ROC AUC, sería interesante evaluar el modelo en función de una matriz de costos (ej. cuánto cuesta un falso positivo vs. un falso negativo en la planta industrial) para definir el umbral óptimo de decisión.