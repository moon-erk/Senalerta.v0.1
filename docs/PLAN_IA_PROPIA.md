# PLAN DE IA PROPIA — SeñAlerta
## Dos modelos entrenados por ti, desarrollados con Claude Code

Este documento explica las dos IAs propias del proyecto, cómo se entrenan, y cómo pedirle a Claude Code que las construya. Está escrito para que entiendas qué hace cada parte y puedas dirigir el desarrollo aunque Claude Code escriba el código.

---

## RESUMEN: QUÉ ES CADA PIEZA

| Pieza | ¿Se entrena? | Qué hace |
|---|---|---|
| MediaPipe | NO (es de Google, cerrado) | Detecta landmarks de manos, cara y cuerpo. Es la ENTRADA de datos. |
| IA #1 — LSTM de señas | SÍ (ya existe, se potencia) | Reconoce qué seña se hace a partir de los landmarks. |
| IA #2 — Clasificador de emoción | SÍ (nuevo) | Predice la emoción facial a partir de los landmarks de la cara. |

Las dos IAs usan los landmarks de MediaPipe como entrada. MediaPipe no se toca — solo se aprovecha mejor.

---

## IA #1 — POTENCIAR EL LSTM DE SEÑAS

### Qué tienes ahora
Una red LSTM que toma 15 frames de 1662 valores cada uno (los landmarks de cuerpo, cara y manos) y predice qué seña es. Funciona, pero se puede hacer más precisa.

### Las cuatro mejoras a aplicar

**Mejora 1 — Aumento de datos (data augmentation)**
El problema: si solo tienes 15 muestras de una seña, el modelo aprende poco. La solución: generar variaciones artificiales de cada muestra para multiplicar los datos sin grabar más. Las variaciones que tienen sentido para señas:
- Pequeñas rotaciones de los puntos (simula que la persona está ligeramente girada)
- Pequeños desplazamientos (simula que está más a la izquierda o derecha)
- Escalado leve (simula que está más cerca o lejos)
- Ruido pequeño en las coordenadas (simula variación natural entre repeticiones)
- Espejo horizontal (una seña hecha con la otra mano)

Con esto, de 15 muestras reales puedes generar 75-150 muestras de entrenamiento. Esto solo ya mejora mucho la precisión.

**Mejora 2 — Mejor normalización**
Ahora los keypoints se normalizan poco. Mejora: normalizar cada frame relativo a un punto de referencia estable del cuerpo (por ejemplo el centro entre los hombros) y a una escala estable (distancia entre hombros). Así la seña se reconoce igual sin importar si la persona está cerca, lejos, a la izquierda o derecha de la cámara.

**Mejora 3 — Arquitectura mejorada**
Probar añadir:
- Capa de normalización por lotes (BatchNormalization) entre capas
- Bidirectional LSTM (lee la secuencia hacia adelante y hacia atrás, capta mejor el movimiento)
- Ajustar el dropout para evitar sobreajuste

**Mejora 4 — Métricas de calidad**
Después de entrenar, generar una matriz de confusión que muestre qué señas se confunden entre sí. Esto te dice exactamente cuáles señas necesitan más muestras o son demasiado parecidas.

### Cómo pedírselo a Claude Code
Prompt sugerido:
"En python/, crea un módulo de aumento de datos llamado augmentation.py que tome una secuencia de keypoints (15 frames de 1662 valores) y genere variaciones: rotación leve, desplazamiento, escalado, ruido y espejo horizontal. Luego modifica training_model.py para que aplique el aumento de datos antes de entrenar, multiplicando cada muestra por 5. Mejora también la normalización en helpers.py para que cada frame se normalice relativo al centro de los hombros y la distancia entre ellos. Después del entrenamiento, genera y guarda una matriz de confusión como imagen. No rompas el formato de 1662 valores ni la compatibilidad con el modelo actual."

---

## IA #2 — CLASIFICADOR DE EMOCIÓN FACIAL

### Qué es
Un modelo nuevo, más simple que el LSTM, que toma los landmarks de la cara de UN frame (no una secuencia) y predice la emoción: feliz, triste, neutral, enojado, sorprendido. A diferencia del LSTM que necesita movimiento, la emoción se lee de una sola foto de la cara.

### Por qué es más simple que el LSTM
Las señas son movimiento (necesitan secuencia de frames → LSTM). La emoción es una postura facial estática (un solo frame → red simple tipo MLP). Por eso este modelo es más fácil y rápido de entrenar.

### Los datos que necesita
Para cada emoción, necesitas ejemplos de los landmarks faciales con esa emoción. Hay dos formas de conseguirlos:
- Grabarlos tú: igual que capturas señas, pero capturando tu cara haciendo cada emoción. Más control, pero requiere grabar.
- Usar un dataset público de emociones faciales (como FER2013) y extraerle los landmarks con MediaPipe. Más datos, pero hay que procesarlos.

Recomendación: empezar grabando tú mismo unas 30-50 muestras por emoción usando la misma infraestructura de captura que ya tienes. Es rápido y suficiente para un primer modelo.

### La arquitectura
Un MLP (perceptrón multicapa) sencillo:
- Entrada: los landmarks faciales (468 puntos × 3 coordenadas = 1404 valores, o un subconjunto de los puntos más expresivos: boca, cejas, ojos)
- Capas: Dense(128) → Dropout → Dense(64) → Dropout → Dense(número de emociones) con softmax
- Salida: probabilidad de cada emoción

### Estructura de datos en disco
Igual que las señas, pero para emociones:
```
AppData/Local/SeñAlerta/
└── emotions/
    ├── feliz/
    │   ├── sample_001.json
    │   └── ...
    ├── triste/
    ├── neutral/
    ├── enojado/
    └── sorprendido/
```
Cada JSON tiene los landmarks faciales de un frame.

### Cómo pedírselo a Claude Code
Prompt sugerido:
"Crea un nuevo sistema de clasificación de emoción facial en python/. Necesito: (1) endpoints en server.py para capturar muestras de emoción (/emotion/capture/start con la emoción como parámetro, /emotion/capture/stop) que guarden los landmarks faciales de cada frame en AppData/SeñAlerta/emotions/[emocion]/sample_XXX.json. (2) Un módulo emotion_model.py con un MLP: Dense(128)→Dropout→Dense(64)→Dropout→Dense(n_emociones)→softmax, que toma los landmarks faciales como entrada. (3) Un endpoint /emotion/train que entrena el modelo con las muestras guardadas. (4) Integrar la predicción de emoción en el loop de captura para que /camera/frame devuelva la emoción detectada en tiempo real. Usa el mismo entorno C:\\v310\\Scripts\\python.exe. No rompas nada del reconocimiento de señas existente."

---

## ORDEN DE DESARROLLO CON CLAUDE CODE

Fase 1 — Potenciar el LSTM (mejora lo que ya funciona)
1. Crear augmentation.py (aumento de datos)
2. Mejorar la normalización en helpers.py
3. Integrar aumento de datos en el entrenamiento
4. Añadir matriz de confusión
5. Reentrenar y comparar precisión antes/después

Fase 2 — Clasificador de emoción (lo nuevo)
6. Endpoints de captura de emoción
7. Capturar muestras de cada emoción (lo haces tú frente a la cámara)
8. Crear emotion_model.py
9. Endpoint de entrenamiento de emoción
10. Integrar predicción de emoción en tiempo real
11. Conectar la emoción al popup de prueba para verla funcionando

Fase 3 — Integración final
12. Las dos IAs corriendo juntas en el mismo backend
13. Verificar que el rendimiento (FPS) sigue siendo bueno con ambos modelos activos

---

## CONSIDERACIÓN IMPORTANTE DE RENDIMIENTO

Correr dos modelos a la vez (LSTM de señas + MLP de emoción) más MediaPipe puede bajar los FPS. Por eso:
- El MLP de emoción es ligero, así que el impacto es pequeño
- La emoción no necesita predecirse en cada frame — basta cada 5-10 frames
- Si los FPS bajan mucho, predecir emoción en un thread separado

---

## QUÉ APRENDES Y PUEDES PRESENTAR DEL PROYECTO

Al terminar tendrás dos IAs entrenadas por ti:
1. Un reconocedor de lengua de señas (LSTM con aumento de datos)
2. Un clasificador de emoción facial (MLP)

Ambos entrenados con tus propios datos, corriendo localmente, sin depender de servicios externos. Eso es un proyecto de machine learning completo y presentable: captura de datos, entrenamiento, evaluación con métricas, e integración en una aplicación real.

---

## NOTA SOBRE MEDIAPIPE (para que no haya confusión)

MediaPipe NO se entrena ni se mejora — es una herramienta cerrada de Google que detecta landmarks. Cuando alguien dice "mejorar la precisión de MediaPipe" en realidad se refiere a mejorar el modelo que USA esos landmarks (tu LSTM). MediaPipe es la cámara que ve los puntos; tu IA es la que interpreta qué significan. La precisión que controlas está en tu IA, no en MediaPipe.
