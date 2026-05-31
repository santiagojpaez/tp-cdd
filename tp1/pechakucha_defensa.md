# Presentación Pechakucha — Defensa TP Ciencia de Datos

> **Formato:** 20 diapositivas × 20 segundos = 6:40 minutos  
> **Enfoque:** Justificación de decisiones. No repetir pasos del informe escrito.  
> **Eje transversal:** Recall como métrica prioritaria — perder una falla es más costoso que una falsa alarma.

---

## Diapositiva 1 — Portada

**Título:** Detectar Antes de Que Sea Tarde — Mantenimiento Predictivo Industrial

**Subtítulo:** Trabajo Práctico N°1 · Ciencia de Datos · UTN FRSF · 2026

**Autores:** Bernard · Brunas · Paez · Paggi

**Visual:** Máquina industrial con panel de alerta. Una pantalla muestra "PREDICCIÓN: FALLA INMINENTE".

---

## Diapositiva 2 — La Decisión Que Importa

**Título:** En Una Planta, Lo Peor Es Lo Imprevisto

**Contenido:**

| Error | Qué pasa en la planta | Costo real |
|-------|----------------------|------------|
| 🔴 **Falso negativo** | El modelo dice "todo bien", la máquina falla | **Parada no programada.** Producción frenada sin aviso. Posible rotura en cadena. Equipo de emergencia. |
| 🟡 Falso positivo | El modelo dice "falla", pero está todo bien | Parada programada. Inspección. Se vuelve a arrancar. Costo menor y controlado. |

> **Decisión metodológica:** La métrica prioritaria en este análisis es **recall** (exhaustividad). Queremos capturar la mayor cantidad posible de fallas reales, aunque eso implique aceptar algunas falsas alarmas. Una parada planificada es un costo; una rotura imprevista es un desastre.

**Visual:** Dos cronogramas de producción: arriba (parada planificada, ordenada), abajo (parada de emergencia, caótica).

---

## Diapositiva 3 — El Problema y Los Datos

**Título:** 7 Sensores, Relaciones Conocidas, Una Variable Objetivo

**Contenido:**

No son números sueltos: las relaciones entre variables responden a leyes físicas.

| Relación | Coef. | Significado |
|----------|:---:|-------------|
| `speed` ↔ `torque` | −0.85 | A velocidad constante, más velocidad ⇒ menos torque |
| `air_temp` ↔ `process_temp` | +0.86 | Intercambio térmico continuo entre ambiente y proceso |
| `tool_wear` vs `target` | — | A mayor desgaste, más fallas |

> **Decisión clave del EDA:** Conservar ambas temperaturas. El gradiente entre ellas (~10 K) tiene significado físico. Validamos: eliminar cualquiera de las dos **degrada** el recall de los modelos.

14.521 observaciones. Target casi balanceado: 51.5% fallas. El detalle completo del análisis exploratorio está en el informe escrito.

**Visual:** Heatmap de correlación con los dos pares destacados + boxplot de tool_wear estratificado por target.

---

## Diapositiva 4 — Eje 1: ¿Por Qué una Red Neuronal?

**Título:** Si El Problema No Es Lineal, Probemos Una Herramienta No Lineal

**Contenido:**

| Lo que sabemos | Implicancia para el modelado |
|----------------|------------------------------|
| Frontera de decisión no lineal | Regresión Logística limitada a recall 0.84 |
| Interacciones de 3+ variables (velocidad × torque × desgaste) | Un modelo debe capturar combinaciones complejas |
| 7 features, ~10k muestras | Territorio favorable para árboles; las redes necesitan más datos |

> **Decisión:** Construir el MLP para comparar. No asumimos que va a ganar — asumimos que la comparación en sí misma es valiosa. Si la red no supera a los ensembles en **recall**, ese resultado negativo también es un hallazgo.

El MLP se diseñó para maximizar capacidad de detección (recall), no precisión bruta. La arquitectura lo refleja: dropout moderado, early stopping, optimizador adaptativo.

**Visual:** Gráfico de dispersión speed vs torque coloreado por target. La zona de falla forma una "L" que ningún hiperplano puede separar limpiamente.

---

## Diapositiva 5 — Eje 1: Diseño de la Arquitectura

**Título:** 64 → 32 → 16: Una Red Que Se Enfoca

**Contenido:**

```
Input (7 features) → [64] → [32] → [16] → Output (1, Sigmoid)
```

| Decisión | ¿Por qué? |
|----------|-----------|
| 3 capas ocultas | Suficientes para modelar interacciones no lineales. Con 2 capas perdemos ~1.3 pp de F1 en validación (evidencia del grid search) |
| 64 neuronas en 1ª capa | ~9× las 7 features. Heurística estándar: ≥2× features para capturar combinaciones. Con 128 violamos el ratio muestras/parámetros |
| 3.361 parámetros | ~8.400 muestras de tuneo → ratio ~2.5:1. Ya es ajustado. Más parámetros = overfitting garantizado |
| Sigmoid en salida | Produce P(falla) ∈ [0,1]. Fundamental para ajustar el umbral de decisión según el costo relativo de cada error |

> Lo que **no** hicimos: redes más profundas o anchas. Con solo 7 features, hay un techo de complejidad que los datos pueden soportar. El informe de ajuste de hiperparámetros detalla esta decisión.

**Visual:** Diagrama de embudo con cantidad de parámetros en cada capa + fórmula del ratio muestras/parámetros.

---

## Diapositiva 6 — Eje 1: Regularización

**Título:** ¿Por Qué Dropout 0.3?

**Contenido:**

Probamos dropout 0.3 vs 0.5 en condiciones idénticas. Resultado:

| Dropout | Rendimiento | ¿Por qué? |
|:---:|------|------|
| **0.3** | **Superior** | Ganó 35 de 36 comparaciones directas |
| 0.5 | Inferior | Sobre-regulariza: solo 32 de 64 neuronas activas en promedio |

> **Decisión:** Rechazamos la recomendación canónica de 0.5 (Srivastava et al., 2014). Esa regla se derivó de redes con >1M de parámetros. Con 3.361, apagar la mitad mutila la capacidad de la red para aprender patrones de falla. Queremos recall alto — no podemos darnos el lujo de perder capacidad expresiva.

**Early Stopping (patience=15):** El modelo decide cuándo parar. Sin esto, configuraciones con lr=0.01 habrían seguido 200 épocas sobreajustando. Ahorró ~60% de cómputo y preservó el mejor recall de validación.

**Visual:** Curvas de pérdida train/val superpuestas con el punto de early stopping resaltado.

---

## Diapositiva 7 — Eje 1: Hiperparámetros

**Título:** 72 Combinaciones, Una Prioridad: Detectar Fallas

**Contenido:**

Grid search sobre validación (1.479 muestras, 15% del train):

| Hiperparámetro | Valores | Criterio de selección |
|---------------|---------|----------------------|
| Arquitectura | [32,16], [64,32], [64,32,16] | Potencias de 2. Cubre desde sub-capacidad hasta el borde del overfitting |
| Learning rate | 0.01, 0.001, 0.0005 | Escala logarítmica. Adam tolera 0.01; SGD necesita ≤0.001 |
| Dropout | 0.3, 0.5 | 0.3 regulariza sin mutilar; 0.5 es el canónico para redes grandes |
| Batch size | 32, 64 | ~260 vs ~130 updates/época. Más ruido con 32 no aportó regularización extra |
| Optimizador | Adam, SGD+momentum | SGD requiere scheduling para competir; Adam converge en menos épocas |

**Configuración ganadora:** [64,32,16] · lr=0.01 · dropout=0.3 · batch=64 · Adam → Val F1=0.9533

**Patrón revelador:** Adam domina el top 5 completo; dropout 0.3 gana sistemáticamente; 3 capas > 2 capas en todos los pares comparables.

**Visual:** Heatmap arquitectura × learning rate coloreado por F1 de validación.

---

## Diapositiva 8 — Eje 1: ¿Qué Detecta el MLP?

**Título:** El MLP Atrapa 96.5% de las Fallas

**Contenido:**

| Métrica | MLP (test) |
|---------|:---:|
| **Recall ⭐** | **0.965** |
| Precisión | 0.919 |
| F1 | 0.941 |
| AUC | 0.987 |

**Matriz de Confusión:**

| Real \ Predicho | Normal | Falla |
|-----------------|-------:|------:|
| Normal | 1.933 | 179 |
| Falla | **75** | **2.037** |

> **Lectura enfocada en recall:** De 2.112 fallas reales, el MLP solo deja pasar **75** (3.5%). Las otras 2.037 fueron detectadas a tiempo. El costo de esas 179 falsas alarmas es aceptable frente al beneficio de haber anticipado 2.037 fallas reales.

**Visual:** Matriz de confusión con las 75 fallas no detectadas en rojo (pocas) y las 2.037 detectadas en verde (muchas).

---

## Diapositiva 9 — Eje 2: La Decisión Silenciosa Que Lo Cambió Todo

**Título:** `Normalizer()` vs `StandardScaler()` — No Es Cosmética

**Contenido:**

| | Normalizer (original, erróneo) | StandardScaler (corregido) |
|---|---|---|
| Operación | Cada **fila** → norma L2 = 1 | Cada **columna** → μ=0, σ=1 |
| Efecto en datos | Proyecta muestras a una hiperesfera | Centra y escala preservando identidad de cada sensor |
| Consecuencia en modelos | Distorsiona distancias (KNN), confunde coeficientes (LR), caotiza gradientes (MLP) | Comportamiento esperado para todos los algoritmos |

> **Decisión:** Corregir **antes** de entrenar el MLP. Las redes neuronales asumen features con μ≈0 y σ≈1. Sin esto, los gradientes de `speed` (~1500 RPM) dominan a los de `air_temp` (~300 K) por un factor 5×. La red aprende a predecir fallas mirando RPM e ignorando las temperaturas.

**Visual:** Dos paneles comparativos: arriba, nube de puntos comprimida en un círculo (Normalizer); abajo, nube centrada y expandida (StandardScaler).

---

## Diapositiva 10 — Eje 2: El Impacto Medido

**Título:** Corrección Validada: Lo Que Cambió en los Modelos

**Contenido:**

Re-ejecución completa con StandardScaler:

| Modelo | Δ Recall | Interpretación |
|--------|:---:|-------|
| Regresión Logística | **+5 pp** | Un modelo lineal no puede compensar features mal escaladas. Es el más sensible |
| Naive Bayes | −3 pp | La normalización L2 enmascaraba correlaciones. Al exponerlas, NB sufre |
| KNN, Árboles, Ensembles | +1 a +2 pp | Incluso modelos "invariantes a escala" mejoran sus splits con datos correctos |

> **Decisión validada:** Esta corrección fue la de mayor impacto individual en todo el proyecto. Sin ella, cualquier ranking de modelos —y cualquier conclusión sobre qué modelo detecta más fallas— habría estado viciado desde la raíz.

**Visual:** Gráfico de barras comparativas: "antes" (rojo, más bajo) vs "después" (verde, más alto) para cada modelo.

---

## Diapositiva 11 — Eje 2: El MLP Sin Estandarización

**Título:** Sin StandardScaler, Esta Red No Detecta Fallas

**Contenido:**

¿Qué hubiera pasado si entrenábamos el MLP con datos del Normalizer?

| Problema | Consecuencia en el MLP |
|----------|------------------------|
| `speed` ~1500 RPM vs `air_temp` ~300 K | El gradiente de `speed` es 5× mayor. La red aprende a predecir solo con RPM |
| Features no centradas en 0 | Inicialización Xavier/He asume μ=0. Pesos iniciales nacen desequilibrados |
| Proyección a hiperesfera | Las correlaciones reales (speed↔torque, −0.85) se distorsionan artificialmente |

> Con StandardScaler: convergencia limpia en 59 épocas, gap train/val = 0.12, recall final 0.965. Sin estandarización: divergencia temprana, la red habría convergido a un mínimo local donde solo "ve" la variable de mayor escala. **El recall se habría desplomado.**

BatchNorm dentro de la red ayudó, pero no reemplaza la estandarización previa. Actúa sobre activaciones, no sobre los features de entrada.

**Visual:** Curvas de convergencia: izquierda (simulación sin estandarizar, divergencia), derecha (datos reales, convergencia suave con recall estable).

---

## Diapositiva 12 — Eje 3: Comparación por Recall

**Título:** El Ranking Que Importa: ¿Cuántas Fallas Detecta Cada Modelo?

**Contenido:**

| # | Modelo | **Recall ⭐** | Fallas perdidas (de 2.112) | AUC |
|---|--------|:---:|:---:|:---:|
| 🥇 | **KNN (k=3, Manhattan)** | **0.985** | **32** | 0.978 |
| 🥈 | Random Forest | 0.983 | 36 | 0.996 |
| 🥈 | Gradient Boosting | 0.983 | 37 | 0.996 |
| 4 | **MLP (PyTorch)** | **0.965** | **75** | 0.987 |
| 5 | Árbol Decisión | 0.956 | 93 | 0.952 |
| 6 | Naive Bayes | 0.890 | 233 | 0.888 |
| 7 | Regresión Logística | 0.843 | 332 | 0.919 |

> **Lectura:** KNN es el campeón en recall: solo 32 fallas no detectadas (1.5%). GB y RF le siguen de cerca (36-37 fallas perdidas). El MLP, con 75 fallas perdidas (3.5%), es competitivo y supera ampliamente a los modelos lineales.

Los valores de precision, F1 y accuracy están en el informe. Acá destacamos la métrica que define el costo operativo real.

**Visual:** Barras horizontales ordenadas por recall + número de fallas perdidas al lado de cada barra.

---

## Diapositiva 13 — Eje 3: ¿Por Qué KNN Lidera en Recall?

**Título:** Vecinos Cercanos, Decisiones Locales

**Contenido:**

KNN con k=3 y métrica Manhattan alcanza recall 0.985. ¿Por qué?

| Propiedad de KNN | Cómo ayuda al recall |
|------------------|---------------------|
| Decisión por vecindad local | Si 2 de 3 vecinos más cercanos son fallas, predice falla. No necesita aprender una frontera global |
| Sin entrenamiento explícito | No hay riesgo de "olvidar" casos raros. Todos los ejemplos de falla están disponibles en inferencia |
| Métrica Manhattan | Penaliza diferencias por feature de forma aditiva. Con 7 features estandarizados, cada sensor pesa lo que debe |

> **Costo:** KNN es lento en inferencia (O(n) por predicción, 766 KB en memoria). Para una planta con miles de predicciones por minuto, esto puede ser prohibitivo. Pero si el objetivo es **maximizar recall sin importar el costo computacional**, KNN es imbatible.

**Visual:** Diagrama de KNN: punto de consulta rodeado de 3 vecinos (2 rojos = falla, 1 verde = normal → predice falla).

---

## Diapositiva 14 — Eje 3: ¿Dónde Queda el MLP?

**Título:** La Red Neuronal en Contexto

**Contenido:**

| Comparación | MLP vs ... | Resultado |
|-------------|-----------|-----------|
| MLP vs Lineales (LR, NB) | recall 0.965 vs 0.84-0.89 | ✅ **Amplia ventaja** (+7 a +12 pp) |
| MLP vs Árbol simple (DT) | recall 0.965 vs 0.956 | ✅ **Ligera ventaja** (+1 pp) |
| MLP vs KNN | recall 0.965 vs 0.985 | ❌ Por debajo (−2 pp) |
| MLP vs Ensembles (RF, GB) | recall 0.965 vs 0.983 | ❌ Por debajo (−1.8 pp) |

> **Interpretación:** El MLP no es el mejor, pero **está en la pelea**. Con recall 0.965, detecta el 96.5% de las fallas. La diferencia con los líderes (KNN, GB, RF) es de 32-38 fallas sobre 2.112 — significativa pero no abismal. Donde el MLP sí falla es en el costo de entrenamiento: 54.7s vs 0.7s de KNN o 34.7s de GB.

**Visual:** Barras de recall ordenadas de mayor a menor, con una línea horizontal en 0.965 (MLP) destacada.

---

## Diapositiva 15 — Eje 3: ¿Puede el MLP Mejorar su Recall?

**Título:** Ajustando el Umbral de Decisión

**Contenido:**

El MLP produce P(falla). El umbral por defecto es 0.5. Si bajamos el umbral, priorizamos recall sobre precisión:

| Umbral | Recall | Precisión | Fallas perdidas | Falsas alarmas |
|:---:|:---:|:---:|:---:|:---:|
| 0.50 | 0.965 | 0.919 | 75 | 179 |
| **0.40** | **~0.978** | ~0.88 | ~47 | ~270 |
| 0.30 | ~0.990 | ~0.82 | ~21 | ~420 |
| 0.20 | ~0.995 | ~0.75 | ~11 | ~600 |

> **Decisión:** El umbral óptimo depende del costo relativo. Si una falla no detectada cuesta 10× más que una falsa alarma, umbral 0.30 es razonable. Si cuestan parecido, mantener 0.50. La ventaja del MLP es que **no requiere reentrenar** para ajustar este balance — basta cambiar el threshold en inferencia.

**Visual:** Curva Precision-Recall del MLP con 4 umbrales anotados. La zona de alto recall (>0.97) está destacada.

---

## Diapositiva 16 — Eje 3: Veredicto

**Título:** ¿Justifica la Complejidad del MLP?

**Contenido:**

| Pregunta | Respuesta |
|----------|-----------|
| ¿Supera en recall a modelos lineales? | ✅ Sí (+12 pp sobre LR) |
| ¿Supera en recall a un árbol simple? | ✅ Sí (+1 pp sobre DT) |
| ¿Supera en recall a los ensembles? | ❌ No (−1.8 pp vs GB/RF) |
| ¿Es más rápido que las alternativas? | ❌ No (el más lento: 54.7s) |
| ¿Es interpretable? | ❌ No — no podemos explicar por qué predice falla |
| ¿Escalaría mejor con más datos? | ✅ Posiblemente — pero no con este dataset |

> **Conclusión defendible:** Para maximizar recall sobre 7 features tabulares con ~10k muestras, **KNN** (0.985) y **Gradient Boosting** (0.983) son superiores al MLP (0.965). La red neuronal es competitiva y supera a los modelos lineales, pero no justifica su costo adicional de entrenamiento y su opacidad cuando hay alternativas más simples con mejor rendimiento.

**Visual:** Gráfico radar de 5 dimensiones comparando MLP vs KNN vs GB: recall, velocidad, interpretabilidad, memoria, escalabilidad.

---

## Diapositiva 17 — Eje 4: Elección para Planta Industrial

**Título:** ¿Qué Modelo Ponemos en Producción?

**Contenido:**

Criterios ponderados para entorno real:

| Criterio | Peso | Justificación |
|----------|:---:|-------|
| **Recall** (no perder fallas) | ⭐⭐⭐⭐⭐ | Una falla no detectada = parada no programada = pérdida máxima |
| Interpretabilidad | ⭐⭐⭐⭐ | El operador necesita confiar y entender la decisión |
| Velocidad de inferencia | ⭐⭐⭐ | En tiempo real, con 7 features no es un cuello de botella |
| Reentrenamiento | ⭐⭐ | Offline, no crítico |

**Matriz de decisión:**

| Modelo | Recall | Interpretable | Velocidad | Puntaje |
|--------|:---:|:---:|:---:|:---:|
| **KNN (k=3)** | 🥇 0.985 | ❌ | ⚠️ Lento | ⭐⭐⭐⭐ |
| **GB** | 🥈 0.983 | ❌ | ✅ | ⭐⭐⭐⭐ |
| **DT** | 0.956 | 🥇 | ✅ | ⭐⭐⭐ |
| MLP | 0.965 | ❌ | ✅ | ⭐⭐ |

**Visual:** Tabla de puntaje ponderado con heatmap. KNN y GB destacados.

---

## Diapositiva 18 — Eje 4: El Dilema Final

**Título:** Máximo Recall vs Máxima Confianza

**Contenido:**

Dos escenarios en el turno noche:

**Opción A — KNN (recall 0.985):**
> Solo 32 fallas no detectadas de 2.112. El precio: 162 falsas alarmas y 766 KB en memoria. Inferencia lenta.
> Si cada falla no detectada cuesta $50.000 y cada falsa alarma $5.000 → costo total: 32×50k + 162×5k = **$2.41M**

**Opción B — Gradient Boosting (recall 0.983, AUC 0.996):**
> 37 fallas no detectadas, 84 falsas alarmas. Inferencia instantánea, 860 KB.
> Mismo cálculo: 37×50k + 84×5k = **$2.27M**

> Con estos números, GB gana por margen estrecho. Pero el punto no es el número exacto — es que **la diferencia en recall entre el mejor y el tercero son solo 22 fallas sobre 2.112**. Lo que realmente separa a los modelos es el costo operativo total (falsos negativos + falsos positivos), no una métrica aislada.

**Visual:** Dos columnas comparativas con KNN vs GB: recall, falsas alarmas, costo simulado.

---

## Diapositiva 19 — Eje 4: Lo Que Aprendimos

**Título:** 5 Decisiones Que Definen el Proyecto

**Contenido:**

| # | Decisión |
|---|----------|
| 1 | **Recall sobre F1.** En mantenimiento predictivo, perder una falla es más caro que una falsa alarma. Esto reordenó nuestro ranking de modelos. |
| 2 | **StandardScaler sobre Normalizer.** La corrección de mayor impacto. Sin ella, ningún modelo —y especialmente el MLP— habría alcanzado su potencial de detección. |
| 3 | **Dropout 0.3 sobre 0.5.** La recomendación canónica asume redes grandes. Con 3.361 parámetros, menos es más. Validado en 35 de 36 comparaciones. |
| 4 | **KNN sorprende.** Con k=3 y Manhattan, detecta el 98.5% de las fallas. Es el rey del recall, aunque paga el precio en velocidad de inferencia. |
| 5 | **El MLP compite.** Recall 0.965, supera a modelos lineales y al árbol simple. Pero con estos datos, no destrona a KNN ni a los ensembles. |

**Visual:** Timeline con los 5 hitos. Cada uno con un ícono representativo.

---

## Diapositiva 20 — Cierre

**Título:** Recomendación Final

**Contenido:**

```
     ¿Qué modelo maximiza la detección de fallas?

  🥇 Gradient Boosting                  🥈 KNN (k=3)
  "Equilibrio óptimo"                   "Máximo recall"

  Recall:     0.983                     Recall:     0.985
  Fallas perdidas: 37/2112              Fallas perdidas: 32/2112
  Falsas alarmas: 84                    Falsas alarmas: 162
  Velocidad:   Instantánea              Velocidad:   Lenta (O(n))
  Memoria:     860 KB                   Memoria:     766 KB
  Interpretable: ❌                      Interpretable: ❌
```

**Si el costo de inferencia no es limitante → KNN.**  
**Si se necesita velocidad + equilibrio → Gradient Boosting.**  
**Si se requiere trazabilidad total → Árbol de Decisión (recall 0.956, completamente auditable).**

El MLP queda como tercera opción: competitivo en recall (0.965), pero sin ventaja decisiva sobre alternativas más simples para este dataset.

**Visual:** Tres cards con los modelos recomendados, ordenados por recall. MLP en gris como mención.

---

*Basado en `tp_cdd.ipynb`, modelo MLP en PyTorch e informe escrito del TP1 (2026).*
