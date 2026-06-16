# El flujo de datos: de la cámara a la palabra reconocida

Este es el camino completo que recorre la información, paso a paso. Cada paso transforma los datos en algo más útil para el siguiente.

```
Cámara → MediaPipe → 1662 keypoints por frame → secuencia de 15 frames → LSTM → predicción (softmax)
```

## Paso 1 — La cámara

La cámara captura video: muchas imágenes por segundo (frames). Cada frame es una foto normal, llena de píxeles. Pero a la red neuronal no le interesan los píxeles: el color de la camiseta o el fondo de la habitación no dicen nada sobre la seña. Lo que importa es **dónde están las manos, la cara y el cuerpo**.

## Paso 2 — MediaPipe: de imagen a puntos

MediaPipe es una herramienta de Google que detecta la posición del cuerpo en una imagen. Nosotros no la entrenamos, solo la usamos. Recibe el frame y devuelve **landmarks**: puntos con coordenadas que marcan partes del cuerpo.

Usamos MediaPipe **Holistic**, que detecta todo a la vez:

| Parte | Puntos | Valores por punto | Total |
|---|---|---|---|
| Cuerpo (pose) | 33 | 4 (x, y, z, visibilidad) | 132 |
| Cara | 468 | 3 (x, y, z) | 1404 |
| Mano izquierda | 21 | 3 (x, y, z) | 63 |
| Mano derecha | 21 | 3 (x, y, z) | 63 |
| **Total** | | | **1662** |

- **x, y** = posición del punto en la imagen (horizontal y vertical)
- **z** = profundidad estimada (qué tan cerca de la cámara)
- **visibilidad** = qué tan seguro está MediaPipe de ver ese punto (solo en el cuerpo)

Si una parte no se ve (por ejemplo, la mano izquierda está fuera de cámara), sus valores se llenan con ceros.

Esto lo hace la función `extract_keypoints` en `python/helpers.py`.

## Paso 3 — El frame se convierte en 1662 números

Cada frame de video queda reducido a una lista de **1662 números**. Eso es todo lo que la red ve: no la imagen, solo el "esqueleto" numérico de la persona en ese instante.

Antes de usarlos, los **normalizamos** (función `normalize_frame` en `helpers.py`): tomamos el punto medio entre los hombros como origen y la distancia entre los hombros como escala. Así la seña se representa igual sin importar si la persona está cerca, lejos, a la izquierda o a la derecha de la cámara. Es como redibujar siempre el esqueleto en el mismo lugar y al mismo tamaño.

## Paso 4 — Se arma la secuencia de 15 frames

Una seña no es un instante, es un movimiento. Por eso agrupamos **15 frames seguidos** (la constante `MODEL_FRAMES = 15` en `constants.py`).

El resultado es una tabla de 15 filas × 1662 columnas: la "mini-película" numérica de la seña.

- Si la grabación tiene menos de 15 frames, se interpola (se inventan frames intermedios) o se rellena.
- Si tiene más, se seleccionan 15 repartidos.
- Eso lo hace `normalize_keypoints` en `helpers.py`.

## Paso 5 — La LSTM procesa la secuencia

La red LSTM (ver [[que-es-el-lstm]]) lee los 15 frames en orden, frame por frame, acumulando en su memoria la "idea" del movimiento. Al final produce un resumen interno de toda la seña. La arquitectura exacta está en [[arquitectura-del-modelo]].

## Paso 6 — Softmax: la predicción final

La última capa de la red usa **softmax**, que convierte el resultado en porcentajes: una probabilidad para cada seña que el modelo conoce, y todas suman 100%.

Ejemplo con 3 señas conocidas:

```
hola     →  92%   ← la ganadora
gracias  →   6%
adiós    →   2%
```

La app solo acepta la predicción si la probabilidad ganadora supera el **umbral de confianza de 0.8** (80%, la constante `THRESHOLD`). Si ninguna seña llega al 80%, la app prefiere no decir nada antes que equivocarse.

## El flujo completo, resumido

```
1. Cámara         →  foto (píxeles)
2. MediaPipe      →  1662 números (esqueleto del frame)
3. Normalización  →  esqueleto centrado y a escala fija
4. Acumulación    →  15 frames = una secuencia (15 × 1662)
5. LSTM           →  "entiende" el movimiento completo
6. Softmax        →  porcentaje para cada seña
7. Umbral 80%     →  si supera, la palabra se confirma y se acumula en la frase
```
