# Inciso 2 — Ajuste de Hiperparámetros

> **Contexto del curso:** Este análisis asume que el lector conoce los fundamentos de un Perceptrón Multicapa (MLP): qué son las capas, neuronas, funciones de activación (ReLU, Sigmoide) y el algoritmo de backpropagation. Cada decisión de diseño se justifica desde cero, incluyendo aquellas que en la práctica suelen darse por sentadas.

---

## 0. ¿Qué son los hiperparámetros y por qué los ajustamos?

En una red neuronal hay dos tipos de "números" que determinan su comportamiento:

| Tipo | ¿Quién los determina? | Ejemplos |
|------|----------------------|----------|
| **Parámetros** | El algoritmo de aprendizaje (backpropagation) | Pesos y biases de cada neurona |
| **Hiperparámetros** | El diseñador (nosotros), **antes** de entrenar | Learning rate, número de capas, dropout, batch size |

**Analogía:** Los parámetros son como las habilidades que el modelo *aprende* practicando. Los hiperparámetros son como el *método de estudio*: ¿cuántas horas por día? ¿con qué intensidad? ¿cada cuánto descansa? Un mal método de estudio arruina incluso al estudiante más capaz. De la misma forma, una mala elección de hiperparámetros puede hacer que incluso una arquitectura bien diseñada no aprenda nada útil.

**¿Por qué no existe una "fórmula mágica"?** Porque la combinación óptima depende de:
- El **tamaño del dataset** (más datos → se pueden usar redes más grandes)
- La **complejidad del problema** (más features o relaciones no lineales → más capacidad)
- El **ruido en los datos** (más ruido → más regularización)
- La **restricción de cómputo** (menos recursos → arquitecturas más chicas, menos épocas)

Por eso **el ajuste de hiperparámetros es empírico**: probamos combinaciones, medimos resultados, y elegimos la mejor. No hay atajos.

---

## 1. Espacio de Búsqueda Definido

Se exploró un grid de **72 combinaciones** (3 arquitecturas × 3 learning rates × 2 dropouts × 2 batch sizes × 2 optimizadores). Cada combinación se evaluó sobre un conjunto de validación (15% del train, 1479 muestras), usando early stopping con patience=15 y un máximo de 200 épocas.

> **¿Por qué grid search y no otro método?** Para un curso inicial, el grid search tiene una ventaja pedagógica fundamental: explora el espacio de manera **sistemática y predecible**. Cada combinación se prueba exactamente una vez, y podemos comparar "manzanas con manzanas": ¿qué pasa si cambiamos SOLO el learning rate y dejamos todo lo demás igual? Métodos más avanzados como random search o Bayesian optimization (Optuna) son más eficientes pero ocultan el "por qué" de cada mejora, que es justamente lo que queremos aprender en esta etapa.

| Hiperparámetro | Valores explorados | Cantidad |
|---------------|-------------------|:---:|
| Arquitectura (capas ocultas) | [32,16], [64,32], [64,32,16] | 3 |
| Tasa de aprendizaje (lr) | 0.01, 0.001, 0.0005 | 3 |
| Dropout | 0.3, 0.5 | 2 |
| Tamaño de batch | 32, 64 | 2 |
| Optimizador | Adam, SGD (momentum=0.9) | 2 |
| **Total combinaciones** | | **72** |

### 1.1 Justificación detallada de cada rango

#### Arquitecturas: ¿cuántas capas y cuántas neuronas?

> **Regla práctica (Goodfellow, Bengio & Courville, "Deep Learning", Cap. 6):** La cantidad de neuronas en la primera capa oculta debería ser al menos del orden de la cantidad de features de entrada. Una heurística común es empezar con ~2× el número de features, luego reducir progresivamente (patrón "funnel" o embudo).

Nuestro dataset tiene **7 features de entrada** y ~10k muestras. Aplicando la heurística:

| Arquitectura | Neuronas 1ᵃ capa | Relación con 7 features | Justificación |
|-------------|------------------|------------------------|---------------|
| `[32,16]` | 32 | ~4.6× | Línea de base mínima. Red pequeña, rápida de entrenar. Adecuada si el problema tiene poca no linealidad. |
| `[64,32]` | 64 | ~9.1× | Red media. Duplica la capacidad de la anterior en la primera capa, permitiendo capturar más interacciones entre features. |
| `[64,32,16]` | 64 | ~9.1× | Red "profunda" (3 capas). Agrega una tercera capa para mayor capacidad de abstracción. Mayor riesgo de overfitting. |

> **¿Por qué potencias de 2?** Es una convención originada en eficiencia de hardware (las GPUs asignan memoria en bloques alineados a potencias de 2). No es una regla matemática — [16,8] o [50,25] también funcionarían — pero adoptar la convención facilita la comparación con la literatura y evita decisiones arbitrarias sobre números "redondos".

**¿Por qué el patrón funnel (64 → 32 → 16)?** Cada capa sucesiva tiene menos neuronas que la anterior. Esto fuerza a la red a crear representaciones cada vez más **compactas** de la información: la primera capa aprende combinaciones simples de features crudos, la segunda combina esas combinaciones en patrones más abstractos, y la tercera sintetiza todo en una representación de alta densidad. Es análogo a resumir un texto: primero identificás ideas clave (64 conceptos), después las agrupás en temas (32 temas), y finalmente destilás la conclusión (16 ideas fuerza). Si hiciéramos lo opuesto (16 → 32 → 64, patrón "abierto"), las capas posteriores tendrían que "inventar" información que no existe en los datos — una receta para overfitting.

**¿Por qué NO probamos arquitecturas más grandes (ej. [128,64,32])?** Con solo 7 features de entrada, una red de 128 neuronas en la primera capa tendría 128 × 7 = 896 pesos solo en esa capa, para un dataset de ~8k muestras de entrenamiento. La regla general es tener al menos 10× más muestras que parámetros para evitar overfitting severo. Con [64,32,16] tenemos ~3,361 parámetros totales y ~8,375 muestras de tuneo (ratio ~2.5:1), lo cual ya es ajustado. Duplicar la arquitectura llevaría el ratio por debajo de 1:1, garantizando overfitting.

#### Tasas de aprendizaje (learning rate): ¿qué tan rápido "aprendemos"?

El learning rate (η, "eta") controla el tamaño del paso en cada actualización de pesos durante el descenso de gradiente. En cada iteración:

```
peso_nuevo = peso_viejo - η × gradiente
```

> **Regla práctica (Kingma & Ba, 2014 — paper original de Adam):** El valor por defecto recomendado para Adam es η = 0.001. Para SGD, los valores típicos son η = 0.01 a 0.1, pero requieren decay programado.

| lr | Descripción intuitiva | ¿Cuándo conviene? | Riesgo |
|----|-----------------------|-------------------|--------|
| 0.01 | Pasos **grandes**. El peso cambia mucho en cada actualización. | Convergencia rápida, datasets muy grandes donde cada época es costosa. | Si el paso es demasiado grande, "salta" por encima del mínimo sin llegar a asentarse (oscilación). |
| 0.001 | Pasos **moderados**. El estándar para Adam. | Buen balance velocidad/estabilidad. Primera opción a probar. | Puede requerir más épocas que 0.01. |
| 0.0005 | Pasos **pequeños**. El peso cambia muy poco en cada actualización. | Problemas donde la superficie de pérdida es muy "rugosa" y pasos grandes causan inestabilidad. | Convergencia lenta. Puede necesitar el doble o triple de épocas. |

> **¿Por qué exploramos en escala logarítmica (0.01, 0.001, 0.0005) y no lineal (0.001, 0.002, 0.003)?** El efecto del learning rate no es lineal: la diferencia entre 0.01 y 0.001 (factor 10×) es cualitativamente distinta a la diferencia entre 0.001 y 0.002 (factor 2×). En la práctica, los valores se exploran en potencias de 10 o mitades de década (0.01, 0.005, 0.001, 0.0005, 0.0001). Esto está documentado en la guía práctica de Bengio (2012) sobre recomendaciones para entrenar redes profundas.

#### Dropout: ¿cómo evitamos que el modelo "se copie"?

Dropout (Srivastava et al., 2014) es una técnica de regularización: durante el entrenamiento, en cada paso se "apaga" aleatoriamente un porcentaje de neuronas (su salida se fuerza a 0). Esto obliga a la red a no depender de ninguna neurona en particular, creando redundancia — como un equipo donde cualquier miembro puede faltar y el trabajo sigue haciéndose.

> **Regla práctica (Srivastava et al., 2014):** El paper original recomienda dropout 0.5 para capas ocultas y 0.2 para la capa de entrada. 
> 
> **Pero ATENCIÓN:** Esta recomendación es para **redes grandes** (millones de parámetros, como las de visión por computadora). En redes pequeñas como la nuestra (~3,300 parámetros), apagar el 50% de las neuronas es demasiado agresivo: de 64 neuronas en la primera capa, solo 32 quedan activas en promedio, y de 32 en la segunda, solo 16.

| Dropout | % de neuronas activas (en promedio) | ¿Cuándo conviene? |
|---------|-------------------------------------|-------------------|
| 0.3 | 70% | Regularización **moderada**. Adecuada para redes chicas o datasets con poco sobreajuste. |
| 0.5 | 50% | Regularización **fuerte**. Útil para redes grandes (>10k parámetros) o datasets muy chicos donde el overfitting es severo. |

> **¿Por qué no probamos dropout 0.1 o 0.2?** Valores muy bajos de dropout tienen un efecto regularizador casi nulo — equivalen a no usar dropout. La literatura sugiere que dropout < 0.2 no aporta beneficios medibles en la mayoría de los casos (Srivastava et al., 2014, Sección 6.3).

#### Tamaño de batch: ¿cuántos ejemplos miramos antes de actualizar los pesos?

El batch size determina cuántas muestras se procesan antes de calcular el gradiente y actualizar los pesos. No confundir con "época": una época es una pasada completa por TODO el dataset. Si el dataset tiene 8000 muestras y batch=64, cada época tiene 8000/64 = 125 actualizaciones de pesos.

> **Regla práctica (Bengio, 2012; Masters & Luschi, 2018):** Para datasets de tamaño moderado (10k-100k muestras), batch sizes entre 32 y 256 son típicos. Valores más chicos producen gradientes más ruidosos, lo cual actúa como un regularizador implícito. Valores más grandes producen gradientes más precisos pero requieren más memoria.

| Batch size | Actualizaciones por época (~8375 muestras) | Ruido del gradiente | Efecto |
|-----------|-------------------------------------------|---------------------|--------|
| 32 | ~262 | **Alto** — cada gradiente se calcula con solo 32 ejemplos, por lo que la dirección puede variar mucho entre batches. | Actúa como **regularizador implícito**: el ruido evita que el modelo se "estanque" en mínimos locales afilados. |
| 64 | ~131 | **Moderado** — 64 ejemplos dan una estimación más estable de la dirección correcta. | **Balance estándar** para datasets de ~10k muestras. Convergencia más suave. |

> **¿Por qué solo 32 y 64, y no 128 o 256?** Con 128, cada época tendría solo ~65 actualizaciones — el modelo "ve" menos direcciones de gradiente distintas, lo cual puede ralentizar la exploración del espacio de pesos. Con 256, solo ~33 actualizaciones por época. Para nuestro dataset de 8375 muestras de tuneo, batch=32 y 64 son el rango recomendado por la regla de Bengio.

**Diferencia clave entre batch, época e iteración (con ejemplos de nuestro dataset):**

- **Batch:** 64 muestras procesadas juntas. Una "tanda" del dataset.
- **Iteración:** Una actualización de pesos. En nuestro caso, cada iteración procesa 1 batch. Con batch=64, hay 8375/64 ≈ 131 iteraciones por época.
- **Época:** 131 iteraciones = 1 época = el modelo "vio" las 8375 muestras exactamente una vez.

#### Optimizador: ¿cómo actualizamos los pesos?

El optimizador es el algoritmo que decide **cómo** modificar los pesos a partir del gradiente calculado. No es lo mismo que el learning rate: el learning rate es un número; el optimizador es la **estrategia** completa.

| Optimizador | ¿Cómo funciona? (intuitivo) | Ventaja principal | Desventaja principal |
|-------------|----------------------------|-------------------|---------------------|
| **SGD + momentum** | Como una pelota rodando cuesta abajo: la dirección actual se combina con la dirección anterior (inercia, momentum=0.9). | Tiende a encontrar **mínimos más "planos"**, que generalizan mejor. | Muy sensible al learning rate. Típicamente necesita un scheduler que reduzca η durante el entrenamiento. |
| **Adam** (Kingma & Ba, 2014) | SGD "inteligente": además de la inercia, **adapta el learning rate por cada parámetro individualmente** usando estadísticas de los gradientes pasados. Si un peso recibe gradientes grandes, Adam le baja el paso; si recibe gradientes chicos, le sube el paso. | Convergencia rápida. Robusto a la escala de los features (no requiere normalización perfecta). | En algunos problemas (especialmente visión) puede generalizar peor que SGD bien tuneado. |

> **Regla práctica (comunidad PyTorch/TensorFlow):** Adam es el optimizador por defecto para MLPs y datos tabulares. SGD con momentum se prefiere en visión por computadora (CNNs) donde la generalización es crítica y hay presupuesto para tuning extensivo.

**¿Por qué momentum=0.9 para SGD?** El valor 0.9 es el estándar establecido por Sutskever et al. (2013) y usado en prácticamente toda la literatura. Significa que el 90% de la actualización proviene de la dirección acumulada (inercia) y solo el 10% del gradiente actual. Valores típicos: 0.9 (estándar), 0.99 (más inercia), 0.5 (menos inercia).

#### Función de pérdida: ¿cómo medimos "qué tan mal" está el modelo?

Usamos **Binary Cross-Entropy (BCE)** en todo el tuning. No exploramos alternativas (MSE, Hinge) por una razón fundamental:

> **Principio de máxima verosimilitud (Goodfellow et al., "Deep Learning", Cap. 5.5):** Para clasificación binaria con salida Sigmoide, BCE es la función de pérdida **derivada naturalmente** de asumir que las etiquetas siguen una distribución de Bernoulli. En cristiano: BCE es la función matemáticamente correcta para este problema. Usar MSE (error cuadrático medio) para clasificación binaria causa problemas de convergencia porque la superficie de pérdida se "aplana" cuando las predicciones están cerca de 0 o 1, matando el gradiente.

**¿Qué mide BCE?** Para una predicción `p` (probabilidad estimada) y una etiqueta real `y` (0 o 1):

```
BCE = −[y × log(p) + (1−y) × log(1−p)]
```

- Si `y=1` (falla) y el modelo dice `p=0.99` → BCE ≈ 0.01 (muy bajo, buena predicción)
- Si `y=1` (falla) y el modelo dice `p=0.01` → BCE ≈ 4.6 (muy alto, el modelo está muy equivocado)
- La penalización es **asimétrica y severa** para predicciones confiadas pero incorrectas. Esto es exactamente lo que queremos en mantenimiento predictivo: un "falso negativo confiado" (el modelo dice "no falla" con 99% de seguridad pero SÍ falla) es el peor error posible.

---

## 2. Resultados del Tuning

### 2.1 Top 5 configuraciones

| # | Arquitectura | lr | Dropout | Batch | Opt | Val F1 | Val AUC | Época | Tiempo |
|---|-------------|-----|---------|-------|-----|--------|---------|-------|--------|
| 1 | 3 capas [64,32,16] | 0.001 | 0.3 | 64 | Adam | **0.9409** | 0.9802 | 86 | 24.2s |
| 2 | 2 capas [64,32] | 0.001 | 0.3 | 64 | Adam | 0.9317 | 0.9774 | 68 | 16.1s |
| 3 | 2 capas [64,32] | 0.0005 | 0.3 | 64 | Adam | 0.9284 | 0.9760 | 62 | 16.5s |
| 4 | 3 capas [64,32,16] | 0.001 | 0.5 | 64 | Adam | 0.9270 | 0.9753 | 68 | 21.5s |
| 5 | 2 capas [32,16] | 0.0005 | 0.3 | 64 | Adam | 0.9257 | 0.9717 | 50 | 11.0s |

**Patrón inmediato:** TODAS las top 5 usan Adam con batch=64 y dropout=0.3. Esto no es casualidad — es una señal fuerte de que estas tres elecciones son robustas para este dataset.

### 2.2 Peores 3 configuraciones

| # | Arquitectura | lr | Dropout | Batch | Opt | Val F1 | Val AUC | Época |
|---|-------------|-----|---------|-------|-----|--------|---------|-------|
| 72 | 3 capas [64,32,16] | 0.01 | 0.5 | 64 | Adam | 0.8807 | 0.9444 | 13 |
| 71 | 3 capas [64,32,16] | 0.01 | 0.3 | 64 | Adam | 0.9035 | 0.9689 | 22 |
| 70 | 2 capas [32,16] | 0.01 | 0.5 | 64 | Adam | 0.9036 | 0.9569 | 40 |

**Patrón inmediato:** TODAS las peores usan lr=0.01. La peor combinación global junta los tres factores de riesgo: red profunda + lr agresivo + dropout fuerte. Esto demuestra que los hiperparámetros no actúan de forma independiente — **interactúan**.

---

## 3. Análisis del Impacto de Cada Hiperparámetro

### 3.1 Arquitectura: más capas → mejor, pero con rendimientos decrecientes

| Arquitectura | Mejor F1 (promedio top 3) | Tiempo promedio | Parámetros totales |
|-------------|--------------------------|-----------------|-------------------|
| [32,16] | 0.9138 | 12.0s | 1,377 |
| [64,32] | 0.9269 | 15.5s | 2,721 |
| [64,32,16] | 0.9305 | 20.8s | 3,361 |

**Interpretación:**

- **De [32,16] a [64,32]:** +1.31 puntos de F1. La primera capa pasa de 32 a 64 neuronas, es decir, el modelo tiene el **doble de "detectores de patrones" simples** en la primera capa. Con 7 features de entrada, tener solo 32 neuronas limita las combinaciones que la red puede aprender — 64 neuronas duplica ese abanico de posibilidades.

- **De [64,32] a [64,32,16]:** +0.36 puntos de F1. La tercera capa de 16 neuronas aporta muy marginalmente. ¿Por qué? Porque las dos primeras capas (64 → 32) ya capturan suficiente complejidad no lineal para este problema. La tercera capa está refinando representaciones que ya son buenas. Es como pulir una superficie que ya está lisa: el beneficio es real pero mínimo.

- **Costo marginal creciente:** Cada capa adicional aporta MENOS mejora y cuesta MÁS tiempo. Este es un principio general en deep learning: duplicar la profundidad NO duplica la performance. De hecho, arquitecturas excesivamente profundas para problemas simples sufren del problema del **gradiente evanescente** (aunque ReLU y BatchNorm lo mitigan, no lo eliminan por completo).

> **Principio de parsimonia (Navaja de Occam aplicada a ML):** Entre dos modelos con performance similar, elegir el más simple. [64,32] ofrece el mejor trade-off capacidad/costo para este problema. [64,32,16] solo se justifica si cada fracción de punto de F1 es crítica (ej. mantenimiento predictivo donde una falla no detectada cuesta millones).

### 3.2 Tasa de aprendizaje: el hiperparámetro MÁS impactante

| lr | F1 promedio (todas las configs) | AUC promedio | Épocas hasta best | Diagnóstico |
|----|-------------------------------|-------------|-------------------|-------------|
| 0.01 | 0.9073 | 0.9616 | ~25 | **Overfitting** — converge rápido pero diverge |
| 0.001 | **0.9230** | **0.9733** | ~62 | **Balanceado** — mejor performance sostenida |
| 0.0005 | 0.9202 | 0.9717 | ~66 | **Underfitting** — converge lento, no llega al óptimo |

**Análisis de las curvas de aprendizaje (Sección 5.3 del notebook):**

Las curvas train/val loss cuentan tres historias distintas:

| lr | Train Loss final | Val Loss final | Gap train-val | ¿Qué significa? |
|----|-----------------|----------------|---------------|-----------------|
| 0.01 | ~0.22 | ~0.85 | **0.63** 🔴 | El modelo memoriza el train (loss bajo) pero fracasa estrepitosamente en validación (loss alto). Overfitting clásico. |
| 0.001 | ~0.28 | ~0.40 | 0.12 | Train y val loss se mantienen cerca. El modelo aprende patrones que generalizan. **Comportamiento ideal.** |
| 0.0005 | ~0.32 | ~0.42 | 0.10 | Ambos loss bajan, pero muy lentamente. El modelo **no termina de converger** en 66 épocas. Le faltan más iteraciones. |

**¿Por qué lr=0.01 causa overfitting siendo que hace lo mismo que los otros (minimizar pérdida)?**

El problema no es QUE aprende, sino **CÓMO** aprende. Con lr=0.01, cada actualización de pesos es un salto grande. El modelo "rebota" entre distintas configuraciones de pesos sin asentarse en ninguna. Las curvas de validación **oscilan fuertemente** — un síntoma claro de que el modelo está "nervioso", sobrecorrigiendo constantemente. Eventualmente encuentra una configuración que minimiza el train loss pero que no generaliza: aprendió los detalles irrelevantes del train (ruido) en lugar de los patrones reales (señal).

> **Regla práctica (Goodfellow et al., Cap. 8):** Si el learning rate es demasiado alto, la curva de train loss puede mostrar dientes de sierra o diverger. Si es demasiado bajo, la curva desciende de manera lineal muy lenta y puede estancarse antes de llegar al mínimo.

**¿Por qué lr=0.001 es el punto justo? Coincide exactamente con el valor por defecto de Adam en PyTorch.** Kingma & Ba (2014) determinaron este valor empíricamente tras probar Adam en una batería de problemas (MNIST, CIFAR-10, etc.). No es casualidad que funcione aquí: Adam con lr=0.001 es un excelente "punto de partida universal" para MLPs.

### 3.3 Dropout: más regularización NO siempre es mejor

| Dropout | F1 promedio | AUC promedio | Efecto observado |
|---------|------------|-------------|------------------|
| 0.3 | **0.9225** | **0.9734** | Regularización suficiente, mejor performance |
| 0.5 | 0.9122 | 0.9670 | **Sobre-regularización**: ~1 punto menos de F1 |

**Dropout 0.3 superó a 0.5 en prácticamente todas las comparaciones directas (17 de 18 pares misma-arquitectura-mismo-lr).**

**¿Por qué el paper original recomienda 0.5 pero acá 0.3 funciona mejor?** Contexto. El paper de Srivastava et al. (2014) evaluó dropout en:
- MNIST: redes con 2 capas de **1024-1024** neuronas (~1.2M parámetros)
- CIFAR-10: redes con 3 capas convolucionales + 1 densa
- ImageNet: AlexNet con **60M parámetros**

Nuestra red tiene ~3,300 parámetros. Con dropout 0.5 en una capa de 64 neuronas, solo 32 quedan activas por batch. En la siguiente capa de 32, solo 16. Con tan pocas neuronas activas, la capacidad efectiva de la red se reduce drásticamente — el modelo directamente no tiene suficiente "poder de cómputo neuronal" para aprender patrones complejos.

> **Regla práctica derivada:** dropout > 0.3 solo se justifica cuando la red tiene **más de ~50 neuronas por capa** y el overfitting es evidente (gap train-val grande). Para redes chicas, empezar con 0.2-0.3.

**La combinación [64,32,16] + dropout 0.5 + lr 0.01 fue la PEOR global (F1=0.8807).** Esto es un ejemplo perfecto de interacción negativa entre hiperparámetros: la red más profunda necesita MÁS capacidad expresiva, pero dropout 0.5 le quita la mitad en cada paso, y lr 0.01 le impide encontrar buenos pesos. Tres elecciones que individualmente son "agresivas" y juntas son catastróficas.

### 3.4 Optimizador: Adam vs SGD

| Aspecto | Adam | SGD + momentum (0.9) |
|---------|------|---------------------|
| Convergencia | Rápida (30-80 épocas) | Lenta (requiere 100-200+ épocas) |
| Sensibilidad al lr | Baja (adapta por parámetro) | Alta (necesita lr scheduling para ser competitivo) |
| Performance en validación | Superior en 200 épocas | Inferior sin scheduling |
| Uso estándar en datos tabulares | ✅ Recomendado | ⚠️ Poco común |

En nuestro grid, **Adam dominó consistentemente.** Esto es esperable: SGD con momentum brilla en problemas de visión por computadora (CNNs) donde se entrena por cientos de épocas con learning rate scheduling cuidadoso (Wilson et al., 2017 mostraron que SGD con momentum puede igualar o superar a Adam en generalización, pero requiere tuning extensivo). En datos tabulares con ~10k muestras, Adam es la elección pragmática y efectiva.

> **Regla práctica (comunidad PyTorch):** Adam para MLPs y datos tabulares. SGD + momentum + scheduler para CNNs y problemas de visión. Si tenés recursos para tuning extensivo (>500 épocas), SGD puede generalizar marginalmente mejor.

### 3.5 Tamaño de batch

| Batch size | Iteraciones/época | Ruido del gradiente | Tiempo/época |
|-----------|-------------------|---------------------|-------------|
| 32 | ~262 | Mayor (regularizador implícito) | Mayor |
| 64 | ~131 | Menor (gradientes más precisos) | Menor |

En nuestros resultados, **batch=64 fue consistentemente igual o superior a batch=32** para todas las arquitecturas con Adam. La diferencia promedio fue <0.005 en F1 (es decir, despreciable).

**¿Por qué batch=32 no ayudó como regularizador?** Porque dropout 0.3 ya cumple ese rol. El ruido adicional de batches pequeños es redundante cuando la red ya está regularizada. Batch=32 sería más útil en arquitecturas sin dropout o con dropout muy bajo.

> **Regla práctica (Masters & Luschi, 2018):** "No le bajes el learning rate, subile el batch size." Aunque este paper recomienda batches más grandes para aprovechar paralelismo de GPU, la implicancia práctica para datasets chicos es: si tu red ya converge bien con batch=64, bajar a 32 no aporta beneficios medibles.

---

## 4. Metodología de Evaluación: ¿cómo sabemos que los resultados son confiables?

### 4.1 División train/validación/test: el principio fundamental

```
Dataset completo (14,078 muestras)
│
├── Train (70%, 9,854 muestras) ──→ Se usa para ENTRENAR y TUNEAR
│   │
│   ├── Tune (85% del train, 8,375) ──→ Entrenar el modelo con cada combinación
│   │
│   └── Val (15% del train, 1,479)  ──→ Evaluar F1/AUC de cada combinación → elegir la mejor
│
└── Test (30%, 4,224 muestras) ──→ Se USA UNA SOLA VEZ, al final, para la evaluación definitiva
```

Esta división tripartita es el estándar en machine learning por una razón fundamental: **el test set debe ser completamente invisible durante todo el desarrollo del modelo.**

**Analogía del examen final:** El train set es el material de estudio y los ejercicios de práctica. El validation set es un "simulacro de examen" que podés hacer varias veces para ver cómo vas. El test set es el examen final real: lo ves UNA SOLA VEZ, al final del cuatrimestre. Si usás el examen final para practicar (test leakage), tu nota no refleja lo que realmente aprendiste. De la misma forma, si el test set "filtra" información durante el tuning, las métricas finales están artificialmente infladas.

**¿Por qué 85/15 y no 80/20 o 90/10?** Es una convención práctica: con ~10k muestras de train, 15% para validación son 1,479 muestras — suficiente para que el F1 calculado tenga baja varianza (intervalo de confianza estrecho). Con menos muestras de validación, las métricas serían ruidosas y podríamos elegir una configuración "mala" por azar. Con más, reducimos los datos de entrenamiento. 85/15 es el punto medio recomendado para datasets de este tamaño.

**Stratify:** La división mantiene la proporción 50/50 de clases (falla/no falla) en train, validación, y test. Sin stratify, por azar podríamos tener 70% fallas en validación y 30% en train, haciendo que las métricas no sean comparables.

**Semilla fija (SEED=42):** Garantiza que cualquiera que ejecute el notebook obtenga exactamente los mismos resultados. Sin semilla fija, cada ejecución parte el dataset de manera distinta y los resultados no son reproducibles. El número 42 es una tradición de la comunidad (referencia a "The Hitchhiker's Guide to the Galaxy").

### 4.2 Métrica principal: ¿por qué F1 y no accuracy?

**Métrica principal: F1-Score en validación.**

Justificación para este problema (mantenimiento predictivo):

| Error | Significado en planta industrial | Costo |
|-------|----------------------------------|-------|
| **Falso positivo** (predecir falla, no hay falla) | Parada de máquina innecesaria | Tiempo de producción perdido, costo de inspección |
| **Falso negativo** (predecir normal, hay falla) | Falla no detectada | Rotura catastrófica, riesgo de seguridad |

Ambos errores son graves. El F1-Score es la media armónica de precision y recall, por lo que **penaliza ambos tipos de error por igual**. Accuracy sería engañoso: en un dataset balanceado 50/50, accuracy = 0.80 suena "bueno", pero podría significar que el modelo acierta el 100% de las fallas y solo el 60% de las operaciones normales — un sesgo peligroso. F1 fuerza a que precision y recall estén balanceados.

**AUC-ROC se usó como métrica secundaria** para confirmar que las conclusiones no dependen del umbral de clasificación específico (0.5). AUC mide la capacidad discriminativa del modelo independientemente del threshold elegido.

### 4.3 Early Stopping: ¿cuándo dejamos de entrenar?

El early stopping es una técnica de regularización que **detiene el entrenamiento cuando el modelo deja de mejorar en validación**, aunque el train loss siga bajando.

```
Época 1-30:  Train loss ↓, Val loss ↓ → El modelo está APRENDIENDO. Seguimos.
Época 31-50: Train loss ↓, Val loss → → El modelo está MEMORIZANDO. Peligro de overfitting.
Época 51:    Val loss no mejora por 15 épocas consecutivas (patience=15) → DETENER.
             Restaurar pesos de la mejor época (época 35, donde Val loss fue mínimo).
```

**¿Por qué patience=15 y no 5 o 50?** Es un balance:
- patience=5: se detiene muy rápido ante una meseta temporal, perdiendo posibles mejoras futuras.
- patience=50: espera demasiado, gastando cómputo en épocas improductivas (overfitting).
- patience=15: estándar empírico. Le da al modelo 15 épocas para "romper la meseta". Si en 15 intentos no mejora, es poco probable que lo haga después.

> **Regla práctica:** patience entre 10 y 20 épocas funciona bien para la mayoría de los problemas con datasets de tamaño moderado. Para datasets muy grandes (millones de muestras), patience más bajo (5-10) porque cada época es costosa. Para datasets muy chicos, patience más alto (20-30) porque el validation loss es más ruidoso.

---

## 5. Documentación del Efecto de los Hiperparámetros

### 5.1 Tasa de aprendizaje: el que más impacto tiene

Las curvas de aprendizaje (Sección 5.3 del notebook) muestran tres regímenes claramente diferenciados:

| lr | Régimen | Train Loss final | Val Loss final | Gap | Diagnóstico visual |
|----|---------|-----------------|----------------|-----|-------------------|
| 0.01 | Overfitting | ~0.22 | ~0.85 | **0.63** 🔴 | Train loss cae abruptamente (épocas 1-10), val loss **diverge** a partir de época 30. Curvas con oscilaciones bruscas. |
| 0.001 | Balanceado | ~0.28 | ~0.40 | 0.12 | Ambas curvas descienden suavemente, se mantienen cercanas. Val loss toca mínimo en época ~30-40 y luego el gap se abre muy levemente. |
| 0.0005 | Underfitting | ~0.32 | ~0.42 | 0.10 | Ambas curvas bajan **muy lento**. A época 100 aún no se estabilizan del todo. Podrían seguir mejorando con más épocas. |

**Conclusión visual:** lr=0.01 produce la peor divergencia train/val. El gap de 0.63 es inaceptable — el modelo esencialmente "hizo trampa" memorizando el train. lr=0.001 produce el comportamiento más saludable. lr=0.0005 es demasiado conservador para 100 épocas.

**¿Por qué el learning rate tiene TANTO impacto?** Porque es el único hiperparámetro que afecta directamente la magnitud de CADA actualización de pesos. Arquitectura y dropout afectan la *estructura* del modelo; el learning rate afecta el *proceso mismo de aprendizaje*. Un learning rate mal elegido puede arruinar incluso la arquitectura perfecta, mientras que una arquitectura sub-óptima con buen learning rate todavía puede aprender razonablemente bien.

### 5.2 Dropout: efecto moderado pero consistente

- Dropout 0.3 supera a 0.5 en **17 de 18 comparaciones directas** (misma arquitectura, mismo lr, mismo batch, mismo optimizador).
- La diferencia promedio es de ~0.01 en F1 — no es enorme, pero sí sistemática.
- Dropout 0.5 solo fue competitivo en un caso: arquitectura [32,16] con lr=0.0005, batch=32, SGD. En esa configuración específica (red chica, optimizador lento, batch ruidoso), la sobre-regularización del dropout 0.5 se compensa parcialmente con el ruido del batch pequeño.

**Lección:** El dropout "óptimo" depende del tamaño de la red. La recomendación canónica de 0.5 (Srivastava et al., 2014) asume redes con miles de neuronas por capa. Para redes con decenas de neuronas, 0.3 es más adecuado. Esto ilustra por qué **las reglas generales de la literatura deben adaptarse al contexto específico** — justamente para eso sirve el ajuste de hiperparámetros.

### 5.3 Arquitectura: trade-off capacidad vs costo

- Pasar de [32,16] a [64,32]: **+1.31 puntos de F1** a cambio de +29% más tiempo.
- Pasar de [64,32] a [64,32,16]: **+0.36 puntos de F1** a cambio de +34% más tiempo.

**Principio de rendimientos decrecientes:** Cada unidad adicional de capacidad (neurona o capa) aporta MENOS que la anterior. La primera capa de 64 neuronas captura las interacciones más importantes entre los 7 features. La segunda capa de 32 neuronas combina esas interacciones en patrones de nivel medio. La tercera capa de 16 neuronas intenta extraer patrones aún más abstractos... pero para un problema con solo 7 features, el "techo de complejidad" es bajo. Hay un límite a cuánta abstracción se puede extraer de 7 números.

### 5.4 Interacción entre hiperparámetros: el efecto "combo"

Los hiperparámetros no actúan de forma aislada. La peor combinación global lo demuestra:

```
[64,32,16] + lr=0.01 + dropout=0.5 + batch=64 + Adam → F1 = 0.8807
```

Cada factor por separado es "agresivo":
- Red profunda (3 capas): máxima capacidad → más riesgo de overfitting
- lr=0.01: pasos grandes → el modelo no se asienta
- dropout=0.5: solo 50% de neuronas activas → capacidad efectiva muy reducida

**Juntos son catastróficos** porque se potencian: una red profunda con poca capacidad efectiva (por dropout alto) y que además no puede encontrar buenos pesos (por lr alto) es la tormenta perfecta. Este es un ejemplo clásico de por qué el grid search prueba TODAS las combinaciones: los efectos de interacción pueden ser no lineales y sorprendentes.

---

## 6. Mejor Configuración y Justificación

### Configuración ganadora

```
Arquitectura:  [64, 32, 16]  (3 capas ocultas, patrón funnel)
Learning rate: 0.001
Dropout:       0.3
Batch size:    64
Optimizador:   Adam
Épocas:        86 (early stopping automático)
Val F1:        0.9409
Val AUC:       0.9802
```

### Justificación fundamentada de cada elección

1. **3 capas [64,32,16]:** Aunque la mejora sobre 2 capas [64,32] es marginal (+0.36 F1), en mantenimiento predictivo cada falla detectada adicional puede representar un ahorro significativo en costo de reparación o prevención de accidentes. Con 7 features de entrada, la tercera capa de 16 neuronas refina las representaciones aprendidas por las capas anteriores sin caer en overfitting (gracias a la regularización). Si el contexto fuera distinto (ej. un sistema con restricciones de tiempo real), [64,32] sería la elección pragmática por su mejor relación costo/beneficio.

2. **lr=0.001:** Coincide con el valor recomendado por Kingma & Ba (2014) para Adam. Las curvas de aprendizaje muestran el comportamiento más saludable: train y validation loss se mantienen cercanos, convergencia en ~60-80 épocas, y el gap final controlado (0.12). No hay evidencia de oscilación ni de estancamiento prematuro.

3. **Dropout 0.3:** Regularización suficiente para una red de 3,361 parámetros. Dropout 0.5 sobre-regulariza sistemáticamente (ver Sección 3.3). La red es lo suficientemente pequeña como para que apagar el 30% de las neuronas en cada paso ya provea el efecto de ensemble implícito que busca el dropout, sin mutilar la capacidad expresiva del modelo.

4. **Adam sobre SGD:** En el régimen de 200 épocas con early stopping patience=15, Adam converge más rápido y consistentemente a mejores métricas. SGD con momentum requeriría learning rate scheduling (ReduceLROnPlateau o similar) y más épocas para ser competitivo. Para datos tabulares de ~10k muestras, Adam es la elección estándar y nuestros resultados lo confirman.

5. **Batch size 64:** Con ~8,375 muestras de tuneo, batches de 64 proveen 131 actualizaciones de gradiente por época, suficientes para una convergencia estable. Batch 32 no aporta regularización adicional significativa porque dropout 0.3 ya cumple ese rol. La diferencia entre ambos es despreciable (<0.005 F1), por lo que elegimos 64 por su leve ventaja en velocidad.

---

## 7. Lecciones Aprendidas

1. **El learning rate es el hiperparámetro más impactante.** Una mala elección (0.01) arruina incluso la mejor arquitectura. Es el primer hiperparámetro que debe ajustarse: si el lr no funciona, nada funciona. La regla práctica: empezar con el valor por defecto del optimizador (0.001 para Adam) y explorar hacia arriba y hacia abajo en escala logarítmica.

2. **Más capas ≠ siempre mejor.** La tercera capa aporta solo +0.36 F1 a un costo del 34% más de tiempo. Los rendimientos son decrecientes. Para problemas con pocos features (<20) y datasets de tamaño moderado (~10k), 2 capas ocultas suelen ser suficientes. La profundidad adicional se justifica cuando el problema tiene alta dimensionalidad (cientos o miles de features) o relaciones muy no lineales.

3. **La regularización excesiva es contraproducente.** Dropout 0.5 empeoró sistemáticamente los resultados. La lección: la intensidad de la regularización debe ser proporcional al riesgo de overfitting, que a su vez depende del tamaño de la red y del dataset. Con redes chicas, dropout moderado (0.2-0.3).

4. **El early stopping es un mecanismo de regularización en sí mismo.** Sin early stopping, configuraciones con lr=0.01 habrían seguido entrenando innecesariamente por 200 épocas completas. El mecanismo ahorró ~60% de tiempo de cómputo en promedio y evitó que los modelos sobreentrenen. Además, hace innecesario "adivinar" el número correcto de épocas.

5. **La validación separada del test NO es opcional.** Usar el test set para cualquier decisión (incluyendo early stopping) infla artificialmente las métricas. Nuestro tuning usó exclusivamente el 15% del train como validación. El test set permaneció intacto para la evaluación final, garantizando que las conclusiones sobre la performance del modelo sean honestas.

6. **Los hiperparámetros interactúan.** La peor combinación no fue "un hiperparámetro malo" sino la suma de tres elecciones agresivas que se potenciaron negativamente. El grid search permite detectar estas interacciones, algo que un ajuste secuencial (uno por uno) pasaría por alto.

---

## 8. Limitaciones y Trabajo Futuro

1. **Regularización L2 (weight_decay):** No se exploró. Adam soporta weight_decay como parámetro, que agrega una penalización proporcional al cuadrado de los pesos (‖w‖²). Esto podría complementar al dropout atacando el overfitting desde otro ángulo: dropout crea redundancia entre neuronas, L2 fuerza a que los pesos sean pequeños (más estables). La combinación dropout + weight_decay es estándar en la práctica.

2. **Learning rate scheduling:** Un scheduler como ReduceLROnPlateau (reduce el lr cuando el val loss se estanca) o CosineAnnealing (decae siguiendo una curva coseno) podría mejorar la convergencia de SGD y potencialmente hacerlo competitivo con Adam. En particular, SGD + momentum + CosineAnnealing es una combinación poderosa para generalización.

3. **Grid search vs búsqueda aleatoria vs bayesiana:** El grid search actual explora 72 puntos. Para espacios de búsqueda más grandes, la búsqueda aleatoria (Bergstra & Bengio, 2012) muestrea combinaciones al azar y suele encontrar mejores configuraciones con menos evaluaciones. Métodos bayesianos como Optuna (Akiba et al., 2019) construyen un modelo probabilístico del espacio de búsqueda y eligen inteligentemente la próxima combinación a probar.

4. **Validación cruzada (K-Fold):** La división tune/val actual usa una sola partición (85/15). Stratified K-Fold (k=5) dividiría el train en 5 partes, entrenando 5 veces con distintas combinaciones de entrenamiento/validación. Esto daría estimaciones más robustas (con desviación estándar) de la performance de cada configuración, eliminando la posibilidad de que un "golpe de suerte" en la partición favorezca a una configuración sobre otra. El costo es 5× más tiempo de cómputo.

5. **Exploración de arquitecturas más diversas:** El patrón funnel es una heurística sólida, pero arquitecturas con "cuellos de botella" (ej. [64, 8, 64] — autoencoder-like) o con el mismo ancho en todas las capas ([64, 64, 64]) podrían comportarse distinto. Para datasets con más features, estas variantes merecen exploración.

---

## Referencias

- **Adam:** Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*.
- **Dropout:** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research, 15*(1), 1929-1958.
- **Guía práctica de hiperparámetros:** Bengio, Y. (2012). Practical Recommendations for Gradient-Based Training of Deep Architectures. En *Neural Networks: Tricks of the Trade* (pp. 437-478). Springer.
- **Batch size y learning rate:** Masters, D., & Luschi, C. (2018). Revisiting Small Batch Training for Deep Neural Networks. *arXiv:1804.07612*.
- **Deep Learning (libro de referencia):** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Capítulos 5 (Maximum Likelihood), 6 (Deep Feedforward Networks), 7 (Regularization), 8 (Optimization).
- **Random search vs grid search:** Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research, 13*, 281-305.
- **SGD momentum:** Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the Importance of Initialization and Momentum in Deep Learning. *ICML 2013*.
