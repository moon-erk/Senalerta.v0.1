# Arquitectura del modelo, capa por capa

Este documento explica el modelo actual de SeñAlerta (definido en `python/model.py`), capa por capa, con cada concepto explicado en sencillo.

## El modelo completo de un vistazo

```
ENTRADA: secuencia de 15 frames × 1662 valores
   │
   ▼
1. Bidirectional LSTM (64)    ← lee la secuencia en ambas direcciones
2. BatchNormalization         ← estabiliza los números
3. Dropout (0.5)              ← evita que memorice
   │
   ▼
4. LSTM (128)                 ← resume todo el movimiento en un vector
5. BatchNormalization         ← estabiliza los números
6. Dropout (0.5)              ← evita que memorice
   │
   ▼
7. Dense (64, relu)           ← combina las pistas encontradas
8. Dense (N, softmax)         ← porcentaje para cada seña
   │
   ▼
SALIDA: probabilidad de cada una de las N señas conocidas
```

> Nota: esta es la versión 2 (mejorada) del modelo. La versión 1 original
> (`LSTM 64 → Dropout → LSTM 128 → Dropout → Dense 64 → Dense 64 → softmax`)
> quedó guardada como comentario en `model.py` por si hay que volver a ella.

## Capa por capa

### 1. Bidirectional LSTM (64)

La primera capa es una LSTM (red con memoria, ver [[que-es-el-lstm]]) con **64 neuronas**, envuelta en un "Bidirectional": en realidad son dos LSTM trabajando a la vez, una lee los 15 frames del primero al último y la otra del último al primero. Sus resultados se combinan.

¿Por qué? Algunas señas se distinguen por cómo **empiezan** y otras por cómo **terminan**. Leyendo en ambas direcciones, la red capta ambas pistas.

Esta capa devuelve un resultado **por cada frame** (`return_sequences=True`): no resume todavía, sino que enriquece cada frame con el contexto de toda la secuencia.

### 2 y 5. BatchNormalization

Durante el entrenamiento, los números que circulan entre capas pueden volverse muy grandes o muy pequeños, y eso hace el aprendizaje lento e inestable. BatchNormalization los **reajusta automáticamente a un rango cómodo** después de cada capa.

Analogía: es como bajar o subir el volumen entre canciones para que todas suenen al mismo nivel — el contenido no cambia, pero es más fácil de procesar.

### 3 y 6. Dropout (0.5)

El **dropout** apaga al azar la mitad de las neuronas (0.5 = 50%) en cada paso del entrenamiento. Cada vez se apagan neuronas distintas.

¿Para qué? Para evitar el **sobreajuste** (overfitting): que el modelo se aprenda los ejemplos de memoria en lugar de entender el patrón general. Si se los memoriza, funciona perfecto con los videos de entrenamiento pero falla con señas nuevas.

Analogía: es como estudiar para un examen tapando la mitad de tus apuntes cada vez — te obliga a entender el tema, porque no puedes confiar en memorizar una sola pista.

Importante: el dropout solo actúa durante el entrenamiento. Cuando la app reconoce señas en tiempo real, todas las neuronas trabajan.

### 4. LSTM (128)

Una segunda LSTM, ahora con **128 neuronas** (más capacidad) y unidireccional. A diferencia de la primera, esta **no devuelve un resultado por frame sino uno solo al final** (`return_sequences=False`): un vector de 128 números que resume el movimiento completo de la seña.

A partir de aquí ya no hay "tiempo": toda la secuencia quedó comprimida en ese resumen.

### 7. Dense (64, relu)

Una capa **densa** (o "totalmente conectada"): cada una de sus 64 neuronas mira TODO el resumen anterior y aprende a combinar las pistas. Es donde el modelo razona algo como "movimiento circular + mano cerca de la cara = probablemente tal seña".

**ReLU** es su función de activación: una regla simple que deja pasar los valores positivos y convierte los negativos en cero. Suena trivial, pero es lo que permite a la red aprender patrones complejos en lugar de solo sumas simples.

### 8. Dense (N, softmax) — la salida

La última capa tiene **una neurona por cada seña conocida** (N cambia según cuántas señas se hayan entrenado). **Softmax** convierte los valores de esas neuronas en porcentajes que suman 100%:

```
hola → 92% | gracias → 6% | adiós → 2%
```

La seña con mayor porcentaje es la predicción, y solo se acepta si supera el umbral de 0.8 (80%).

## Conceptos del entrenamiento (en `training_model.py`)

### ¿Qué es una época (epoch)?

Una época es **una pasada completa por todos los ejemplos de entrenamiento**. El modelo necesita ver los ejemplos muchas veces para aprender, igual que repasar los apuntes varias veces. Está configurado hasta 500 épocas, pero casi nunca llega — ver EarlyStopping.

### ¿Qué es EarlyStopping?

Un vigilante que detiene el entrenamiento cuando ya no hay mejora. Cada época mide la precisión sobre los **datos de validación** (ejemplos que el modelo NO usa para aprender — son su "examen sorpresa"). Si pasa 20 épocas sin mejorar (`patience=20`), se detiene y se queda con la mejor versión que hubo. Evita perder tiempo y evita el sobreajuste.

### ¿Qué es el batch size (16)?

El modelo no estudia los ejemplos de uno en uno ni todos a la vez: los procesa en grupos de 16 y ajusta sus neuronas después de cada grupo. Es un equilibrio entre velocidad y estabilidad del aprendizaje.

### ¿Qué es el optimizador Adam?

Es el algoritmo que decide **cuánto corregir** las neuronas tras cada error. Adam es el estándar actual: ajusta el tamaño de las correcciones automáticamente.

### ¿Qué es la función de pérdida (categorical_crossentropy)?

Es la forma de medir "qué tan equivocado" estuvo el modelo en cada predicción. Castiga más estar muy seguro y equivocado, que dudar. El entrenamiento consiste en minimizar este número.

### Aumento de datos (×5)

Antes de entrenar, cada muestra real genera 5 variaciones artificiales (rotación leve, desplazamiento, escalado, ruido, espejo — en `python/augmentation.py`). Así el modelo ve más variedad sin que grabemos más videos.

### Matriz de confusión

Al terminar el entrenamiento se guarda `confusion_matrix.png` en la carpeta de modelos: una cuadrícula que muestra qué señas se confunden entre sí. La diagonal son los aciertos; cualquier mancha fuera de la diagonal señala dos señas que el modelo confunde (y que probablemente necesitan más muestras). Anota los resultados en [[bitacora-experimentos]].
