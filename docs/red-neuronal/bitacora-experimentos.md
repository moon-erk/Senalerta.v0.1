# Bitácora de experimentos de entrenamiento

Anota aquí **cada entrenamiento** que hagas. Así sabrás qué cambios mejoraron el modelo y cuáles no, y tendrás los datos listos para presentar el proyecto.

## Cómo llenar cada columna

- **Fecha**: día del entrenamiento.
- **Nº señas**: cuántas señas distintas conocía el modelo en ese entrenamiento.
- **Muestras/seña**: cuántas muestras reales tenía cada seña (aprox. si varía).
- **Aumento**: factor de aumento de datos usado (×5 es el actual; 0 = sin aumento).
- **Precisión train**: la accuracy de entrenamiento que imprime `training_model.py`.
- **Precisión val**: la accuracy de validación (la importante — es el "examen sorpresa").
- **Épocas**: en qué época se detuvo el EarlyStopping.
- **Cambios hechos**: qué se modificó respecto al entrenamiento anterior (nuevas señas, más muestras, cambio de arquitectura, etc.).
- **Observaciones**: qué señas se confunden según la matriz de confusión, cómo se siente en la prueba real, etc.

## Tabla de experimentos

| Fecha | Nº señas | Muestras/seña | Aumento | Precisión train | Precisión val | Épocas | Cambios hechos | Observaciones |
|---|---|---|---|---|---|---|---|---|
| 2026-06-11 | 2 (hola, gracias) | ~7 (14 en total) | 0 | 100% | 100% | 21 (early stop) | Línea base sin aumento, para comparar | Val de solo 2 muestras — el 100% no es significativo con tan pocos datos. Matriz de confusión falló (val muy pequeño); ya corregido en training_model.py |
| 2026-06-11 | 2 (hola, gracias) | ~7 (14 reales → 84 con aumento) | ×5 | 100% | 100% | 26 (early stop, mejor época 6) | Primer entrenamiento con aumento ×5 + flip corregido (-x en datos normalizados) | Con solo 2 señas la tarea es trivial: ambas corridas dan 100%. El beneficio del aumento se medirá al entrenar el modelo base (Fase C, ~35 señas). Ojo: el aumento se hace ANTES del split train/val, así que variaciones de una misma muestra caen en ambos sets — la precisión de val sale inflada. Considerar dividir antes de aumentar |
| 2026-06-11 | 2 (hola, gracias) | ~7 (14 reales; train: 12→72 aumentadas, val: 2 reales) | ×5 solo train | 100% | 100% | 21 (early stop, mejor época 1) | Corregida la fuga de datos: el split train/val ahora se hace ANTES del aumento, y el aumento se aplica SOLO a train. La validación contiene únicamente muestras reales nunca vistas | Desde esta fila, la precisión de val es honesta (sin fuga). Con 2 señas sigue dando 100% — el set de val es de solo 2 muestras, así que el número real se verá en la Fase C con más señas y más muestras |
| 2026-06-11 | 2 (hola, gracias) | 12 gracias + 96 hola (108 reales; hola.h5 estaba desactualizado con solo 2 y se regeneró) | ×5 solo train | 99.5% | 90.9% (10/11 reales) | 62 (early stop, mejor época 42) | Consistencia del pipeline: (1) evaluate_model.py ahora aplica normalize_frame; (2) entrenamiento usa normalize_keypoints (interpolación a 15 frames) igual que el vivo, en vez de pad/truncate; (3) EarlyStopping monitorea val_loss con start_from_epoch=15 (val_accuracy se saturaba en la época 1 y restauraba un modelo sin entrenar); (4) class_weight balanceado por el desbalance 96 vs 12 | Prueba del pipeline EN VIVO (mismas funciones del loop de server.py, con muestras grabadas): 5/6 reconocidas con confianza >0.8. La única falla es una muestra de gracias→hola (0.88): "gracias" necesita más muestras (tiene 12 vs 96 de hola). Backend arrancado y verificado: /health OK, modelo cargado, cámara a 8 fps |
| | | | | | | | | |

## Ejemplo de fila llena (borrar cuando tengas datos reales)

| Fecha | Nº señas | Muestras/seña | Aumento | Precisión train | Precisión val | Épocas | Cambios hechos | Observaciones |
|---|---|---|---|---|---|---|---|---|
| 2026-06-10 | 5 | 15 | ×5 | 98.2% | 91.5% | 87 | Primera versión con Bidirectional LSTM | "hola" y "buenos días" se confunden — grabar más muestras |

## Optimización de FPS del backend (2026-06-11)

Diagnóstico por etapa del loop de captura (ms por frame, en esta CPU): predict de confianza 545, MediaPipe 287, JPEG+base64 35, sleep 10, landmarks 5, resto <10. El stream iba a ~3 FPS porque TODO corría en un solo hilo.

Cambios aplicados en server.py:
- **Dos hilos**: capture_loop publica cámara+JPEG al ritmo de la cámara; inference_loop corre MediaPipe + reconocimiento a su propio paso sobre el frame más reciente.
- Predict de confianza solo cada 3 iteraciones de inferencia (con caché del último resultado). Se descartó `__call__` directo: medido MÁS LENTO que .predict en este equipo (853 vs 545 ms).
- JPEG calidad 75→60, codificación única por frame, serialización de landmarks única.
- Sin grabs extra (descartaban 2 de cada 3 frames), sleep 0.01→0.001, stream sin frames duplicados.
- Warmup del modelo al arrancar (el primer predict traza el grafo de TF, >10 s).

Resultado: stream de 3 → **24.6 FPS reales** (loop interno a 30-31 FPS, que es el máximo físico de la cámara — se le pidió 60 y el driver entrega 30). Reconocimiento verificado en vivo: seña "gracias" reconocida 5 veces con confianza >0.8 frente a la cámara. Precisión simulada intacta: 5/6, mismas confianzas.

## Distinción letra/palabra (2026-06-11)

- El modelo base de fábrica será SOLO el abecedario (27 letras, tipo "letter"). Saludos y palabras los añade cada usuario.
- LETRA = postura estática: la captura toma 7 frames (LETTER_SAMPLE_FRAMES) tras 2 frames de estabilidad y se cierra sola; manteniendo la mano salen muestras en serie. PALABRA = movimiento: captura mientras las manos están en el encuadre (igual que siempre).
- El tipo de cada seña queda en models/signs_metadata.json. Señas antiguas sin registro cuentan como "word".
- Las secuencias cortas de letra (5-8 frames) se interpolan a 15 con normalize_keypoints, el mismo camino del tiempo real — verificado con 5, 7, 8 y 30 frames.
- Al grabar el abecedario: meta de 20 muestras por letra con el modo "Grabación en serie" del popup. Después: Entrenar modelo base (keypoints + train) desde el mismo panel; muestra la precisión final (val_acc del /train/status).

## Control de calidad del modelo (2026-06-13)

Herramientas de calidad añadidas al popup, todas con endpoints de SOLO lectura
(no modifican el entrenamiento):
- **GET /model/confusion_matrix** — sirve el PNG ya guardado por el entrenamiento.
- **GET /model/quality** — evalúa el modelo guardado sobre las muestras reales de
  cada seña y devuelve precisión propia y confusiones. OJO: no es validación limpia
  (incluye muestras de train), es un control de calidad para saber qué reforzar.
- **GET /model/status** — foto del modelo: nº señas, tipos, muestras, fecha del
  último entrenamiento (mtime del .keras) y última precisión val. (de /train/status,
  que se reinicia al reiniciar el backend → muestra "no disponible" hasta reentrenar).
- **POST /model/sign_type** — persiste el tipo letra/palabra sin capturar (toggle de
  movimiento).

Lectura de calidad del modelo actual (gracias+hola, control de calidad sobre
todas las muestras): gracias 91.7% (se confunde con hola), hola 99.0%. Confirma lo
que ya sabíamos: gracias necesita más muestras (12 vs 96 de hola).

Decisión: las letras con movimiento (J, Z, Ñ por defecto) se capturan como type
"word" (flujo de movimiento), no como "letter" (postura). El usuario puede marcar
otras letras como movimiento con el toggle.

## Recordatorios

- La precisión que importa es la de **validación** (val). Si train es muy alta pero val es baja, hay sobreajuste (ver [[arquitectura-del-modelo]]).
- Después de cada entrenamiento, revisa la matriz de confusión en `AppData/Local/SeñAlerta/models/confusion_matrix.png`.
- Si una seña se confunde mucho con otra, la solución casi siempre es **grabar más muestras** de esas dos señas.
- Meta del modelo base de fábrica: precisión de validación por encima del **85%**.
