# REPORTE DE ESTADO — SeñAlerta

> Generado el 2026-06-13. Verificado contra el código fuente y los datos reales
> en `AppData/Local/SenAlerta/`, no solo contra la documentación.

---

## 1. RESUMEN EJECUTIVO

SeñAlerta es una aplicación de escritorio y móvil que traduce Lengua de Señas
Colombiana (LSC) a voz, y voz a texto, para que una persona sorda converse cara a
cara con un oyente. El oyente se une por código QR a una página web ligera; los
mensajes viajan en tiempo real por Supabase. El usuario principal es la persona sorda.

**Punto general:** el motor de reconocimiento (la parte técnicamente más difícil)
está construido, optimizado y probado en vivo. Pero todavía no existe el modelo
base de fábrica (el abecedario), no hay frontend real (solo una herramienta de
prueba en HTML), y no se ha empezado Supabase ni la versión móvil.

**Avance estimado hacia una primera versión usable: ~30%.** El núcleo de IA está
sólido; falta casi todo lo que rodea a ese núcleo para convertirlo en un producto.

---

## 2. LO QUE YA TENEMOS (verificado contra el código)

### Backend de reconocimiento — `python/server.py`
Servidor Flask servido con Waitress en `localhost:8765`. Endpoints verificados en
el código:
- Cámara: `/camera/start`, `/camera/stop`, `/camera/frame`, `/camera/stream` (MJPEG)
- Captura: `/capture/start` (acepta `type` letra/palabra), `/capture/stop`,
  `/capture/samples` (devuelve `{count, type}`), `/capture/delete`
- Keypoints/entrenamiento: `/keypoints/create`, `/keypoints/status`, `/train`,
  `/train/status` (expone `train_acc` y `val_acc` al terminar)
- Reconocimiento: `/recognition/sentence`, `/recognition/clear`, `/recognition/words`
- Emoción: `/emotion/capture/start|stop`, `/emotion/samples`, `/emotion/train`,
  `/emotion/train/status`
- Salud: `/health`

El loop de captura está dividido en **dos hilos**: uno publica la cámara + JPEG al
ritmo del hardware, otro corre MediaPipe + reconocimiento sobre el frame más
reciente. El `predict` de confianza se ejecuta cada 3 iteraciones con caché.

**Rendimiento real (PROBADO EN VIVO):** el stream alcanzó **24.6 FPS reales**
(loop interno a 30-31, el máximo físico de la cámara) medidos en esta sesión bajo
buena luz. **Pero el FPS es muy sensible a la iluminación:** en la última medición,
con poca luz, cayó a **1-10 FPS** porque la cámara sube el tiempo de exposición
automáticamente. Esto NO es un problema del código sino de las condiciones de
captura — pero es real y hay que tenerlo en cuenta para demos.

Estado: **PROBADO EN VIVO** (reconocimiento, cámara, stream, FPS).

### Sistema de IA — `python/model.py`, `training_model.py`, `helpers.py`, `augmentation.py`
- **Arquitectura (verificada en `model.py`):** Bidirectional LSTM(64) → BatchNorm →
  Dropout(0.5) → LSTM(128) → BatchNorm → Dropout(0.5) → Dense(64, relu) →
  Dense(N, softmax). La arquitectura v1 original quedó comentada por si hay que revertir.
- **Aumento de datos (`augmentation.py`):** rotación ±10°, desplazamiento ±0.05,
  escalado 0.9-1.1, ruido gaussiano σ=0.01 y espejo horizontal con intercambio de
  manos. Factor ×5. Se corrigió el espejo (`-x` en lugar de `1-x`) para que sea
  coherente con los datos normalizados. **PROBADO** (28/28 pruebas en `test_fase1.py`).
- **Normalización corporal (`normalize_frame`):** centra cada frame en el punto medio
  de los hombros y escala por la distancia entre ellos. Se verificó que se aplica
  de forma **idéntica** en los tres puntos del pipeline (creación de keypoints vía
  entrenamiento, entrenamiento, y evaluación en vivo). Antes `evaluate_model.py`
  evaluaba con keypoints crudos — corregido.
- **Pipeline sin fugas de datos:** el split train/validación se hace ANTES del
  aumento, y el aumento se aplica SOLO a train. La validación contiene únicamente
  muestras reales nunca vistas. Aplicado tanto en `training_model.py` como en el
  endpoint `/train` de `server.py`.
- **`class_weight` balanceado** para compensar el desbalance entre señas con muchas
  y pocas muestras.
- **EarlyStopping sobre `val_loss`** con `start_from_epoch=15` (antes monitoreaba
  `val_accuracy`, que se saturaba en la época 1 con sets pequeños y restauraba un
  modelo sin entrenar — bug corregido).
- **Tratamiento temporal unificado:** tanto el entrenamiento como el tiempo real
  llevan las secuencias a 15 frames con `normalize_keypoints` (interpolación), no
  con pad/truncate. Verificado con secuencias de 5, 7, 8 y 30 frames.

Estado: **PROBADO EN VIVO** parcialmente (reconocimiento real funcionó; las mejoras
de pipeline están verificadas en código y por pruebas automatizadas, no con un
dataset grande que demuestre la ganancia de precisión).

### Datos actuales (verificado en `AppData/Local/SenAlerta/`)
**Esta es la realidad de los datos hoy:**
- **Solo 2 señas grabadas**, ambas tipo PALABRA:
  - `hola`: 96 muestras
  - `gracias`: 12 muestras
- **El abecedario NO está grabado.** Hay una entrada huérfana `"a": "letter"` en
  `signs_metadata.json`, pero **no existe ninguna carpeta de frames para "a"**: viene
  de un `/capture/start` de prueba que no llegó a capturar nada. Letras reales
  grabadas: **0 de 27**.
- **Modelo de señas:** `actions_15.keras` existe (entrenado el 2026-06-11), pero
  está entrenado SOLO con esas 2 señas. La última precisión reportada fue ~90.9% de
  validación, **pero sobre un set de validación de apenas 11 muestras** — no es un
  número confiable.
- **Emoción:** existe `emotion_model.keras` y `emotions.json` lista 4 emociones,
  pero en disco **solo hay 8 muestras de "feliz" y ninguna de las demás**. El modelo
  de emoción es un experimento temprano / stub, **no es funcional** en la práctica.

Estado: **VERIFICADO EN DISCO.** El modelo base de fábrica NO existe todavía.

### Herramienta de prueba — `test_popup.html`
HTML puro standalone (sin build). Permite:
- Encender/apagar cámara y ver el stream con barra de confianza e historial de frase
- Capturar señas manualmente con **selector LETRA/PALABRA** y explicación de la diferencia
- **Modo "Grabación en serie"**: overlay con las 27 letras del abecedario, contador
  por letra, meta de 20 muestras, celebración visual al completar, "siguiente
  pendiente", barra global X/27, y botón "Entrenar modelo base" (habilitado solo
  cuando las 27 lleguen a 20)
- Crear keypoints y entrenar (con barras de progreso)
- Captura y entrenamiento de emoción (UI presente)

Estado: **PROBADO con Playwright** (navegador real): indicador verde, stream
renderiza, selector de tipo funciona, modo serie muestra 27 letras, la petición de
captura de letra envía `type: letter`, botón de entrenar deshabilitado mientras
falten letras. El **cierre automático de muestras de letra** se probó alimentando
el `inference_loop` real con frames grabados (7 muestras de 7 frames en serie), **no
con una mano física frente a la cámara**.

### Documentación y memoria del proyecto — `/docs`
Completa y al día: `CLAUDE.md`, `ARQUITECTURA_FUNCIONAL.md`, `DOCUMENTO_MAESTRO.md`,
`PLAN_IA_PROPIA.md`, `GUIA MAESTRA.md` (los pasos por fases con prompts),
`06-PROGRESO.md`, diagramas SVG, y la carpeta `red-neuronal/` con 4 notas
explicativas + una bitácora de experimentos con el historial de entrenamientos.

Estado: **VERIFICADO.** La documentación es uno de los puntos más fuertes del proyecto.

---

## 3. LO QUE FALTA (en orden de las fases del proyecto)

### Fase C restante — Grabar el abecedario y entrenar el modelo base
Grabar las 27 letras (a–z + ñ), ~20 muestras cada una, con el modo grabación en
serie, y pulsar "Entrenar modelo base".
- **Complejidad: baja** (la herramienta ya está hecha y probada).
- **Depende de: trabajo humano** (grabar frente a la cámara). Claude Code ya no
  tiene nada que construir aquí, salvo ajustar si algo falla al entrenar con 27 clases.

### Fase D — Frontend nuevo con las 7 pantallas
Proyecto React + Vite + React Router + Tauri con sistema de diseño de marca, y las
7 pantallas (Bienvenida, Cuenta, Inicio, Conversar, Aprender, Biblioteca, Ajustes),
conectadas al backend que ya funciona.
- **Complejidad: alta** (es la mayor cantidad de trabajo de construcción pendiente).
- **Depende de: Claude Code**, idealmente con diseños de Stitch como referencia.

### Fase E — Supabase (cuentas, sesiones QR, página del oyente)
Auth de Supabase, canal Realtime para las sesiones QR, y la página web ligera del
oyente (texto + voz del navegador + reconocimiento de voz).
- **Complejidad: alta.**
- **Depende de: ambos.** El equipo crea la cuenta de Supabase y pega las claves;
  Claude Code escribe la integración. Es la pieza que hace "la magia" del producto.

### Fase F — Empaquetado e instalador
Backend Python como sidecar de Tauri con PyInstaller, e instalador de Windows.
- **Complejidad: media-alta** (empaquetar TensorFlow + MediaPipe con PyInstaller
  suele dar problemas).
- **Depende de: Claude Code.**

### Versión móvil (TensorFlow Lite)
App nativa con MediaPipe Tasks + TF Lite y el modelo convertido de `.keras` a `.tflite`.
- **Complejidad: alta** (proyecto casi independiente).
- **Depende de: Claude Code**, pero es claramente lo último de la hoja de ruta.

### Clasificador de emoción facial
El modelo MLP existe en código (`emotion_model.py`) y hay endpoints, pero **no hay
datos reales** (solo 8 muestras de "feliz"). Está planeado, no terminado.
- **Complejidad: media** (la infraestructura existe; falta grabar datos y validar).
- **Depende de: trabajo humano** (grabar emociones) + Claude Code (afinar e integrar).

---

## 4. LO QUE PODEMOS MEJORAR (deuda técnica y riesgos)

### Limitaciones actuales conocidas
- **Solo 2 señas y ningún modelo base.** Hoy el sistema "sabe" decir hola y gracias.
  Todo lo demostrable depende de eso. El producto real empieza a existir cuando esté
  el abecedario.
- **Set de validación minúsculo.** Con 11 muestras de validación, el ~90.9% reportado
  no significa casi nada estadísticamente. La precisión real solo se sabrá con el
  abecedario completo.
- **Desbalance de datos:** `hola` tiene 96 muestras y `gracias` 12. Se mitigó con
  `class_weight`, pero lo correcto es grabar cantidades parecidas por seña.
- **Entrada huérfana en metadatos** (`"a": "letter"` sin frames). Inofensiva, pero
  conviene limpiarla antes de grabar en serio.
- **Modelo de emoción no funcional:** existe el archivo pero está entrenado con datos
  insuficientes. Si se muestra en una demo, dará resultados sin sentido. Mejor no
  exhibirlo hasta tener datos reales.

### Riesgos técnicos
- **Caídas silenciosas del backend.** El proceso murió varias veces sin error en el
  log. La causa más probable era `text_to_speech.py` haciendo `pygame init/quit`
  concurrente desde varios hilos con un archivo compartido; se **blindó** (lock +
  mixer único + archivo temporal por reproducción), pero **no se reprodujo la caída
  de forma controlada**, así que no hay 100% de certeza de que esa fuera la única
  causa. Riesgo medio: vigilar si vuelve a ocurrir.
- **Condición de carrera de la cámara.** `camera_stop` podía liberar la cámara
  mientras el hilo de captura seguía dentro de `cap.read()` (código nativo DSHOW,
  capaz de tumbar el proceso). Se **blindó** con `join` de los hilos antes de
  liberar, pero **tampoco se reprodujo el crash original**. Riesgo bajo-medio.
- **El FPS depende de la cámara y la luz, no del código.** Con buena luz, ~25 FPS;
  con poca luz, 1-10 FPS por la auto-exposición. El techo de 30 FPS es físico de la
  webcam. Para superar 30 haría falta otra cámara.
- **TTS depende de internet** (gTTS llama a Google). Sin conexión, el reconocimiento
  funciona pero no habla. La arquitectura prevé que el oyente use la voz del
  navegador, pero la app de la persona sorda usa gTTS hoy.

### Decisiones de arquitectura que podrían necesitar revisión
- **¿Quién habla con Supabase, el backend Python o el frontend?** Sigue pendiente de
  decidir (anotado en el documento maestro). Lo más razonable es el frontend.
- **Backend local Python en escritorio vs. modelo en el dispositivo en móvil:** son
  dos caminos de inferencia distintos (TensorFlow normal vs. TF Lite) que hay que
  mantener coherentes. Riesgo de divergencia si el formato de keypoints o la
  normalización cambian en uno y no en el otro.

### Atajos tomados que habría que pulir
- La **herramienta de prueba es un solo HTML**, no el frontend real. Sirve para
  desarrollar y demostrar, no es el producto.
- **Warmup del modelo** se hace al arrancar para evitar el primer `predict` lento
  (>10 s por el trazado del grafo). Funciona, pero alarga el arranque a 1-2 minutos.
- Varios endpoints **no validan exhaustivamente** la entrada ni manejan todos los
  errores con elegancia (suficiente para desarrollo, no para producción).

### Consideraciones legales pendientes
- **Datos biométricos (Ley 1581 de Colombia).** La geometría facial es dato sensible.
  La arquitectura ya define la regla correcta (la cara nunca sale del dispositivo;
  solo se donaría el esqueleto de manos/cuerpo, anónimo y con consentimiento), pero
  **nada de eso está implementado todavía** porque no hay Supabase ni donación.
- **Consentimiento informado y menores.** El público incluye instituciones educativas
  con posibles menores. El flujo de consentimiento explícito para donar señas aún no
  existe. Debe construirse ANTES de activar cualquier subida de datos.

---

## 5. RECOMENDACIÓN DE PRÓXIMOS PASOS

**1. Grabar el abecedario y entrenar el modelo base (Fase C, trabajo humano).**
Es lo de menor esfuerzo y mayor impacto: la herramienta ya está lista y probada, y
sin un modelo base el proyecto no tiene nada real que mostrar más allá de dos
palabras. Además, entrenar con 27 clases es la primera prueba de verdad del pipeline
de IA (hasta ahora validado solo con 2 señas). Limpiar antes la entrada huérfana
`"a"` de los metadatos.

**2. Construir el frontend nuevo (Fase D).**
Una vez exista el modelo base, el cuello de botella pasa a ser que no hay producto
visible: solo un HTML de prueba. El frontend con las 7 pantallas es el grueso del
trabajo de construcción pendiente y lo que convierte el motor en una app. Conviene
empezar por el sistema de diseño y la pantalla Conversar (la principal), reusando la
conexión al backend que ya funciona.

**3. Integrar Supabase y las sesiones QR (Fase E).**
Es lo que hace realidad la promesa central del producto: la conversación en vivo
entre la persona sorda y el oyente por QR. Va después del frontend porque necesita
las pantallas donde vivir, pero es la pieza que diferencia a SeñAlerta de un simple
reconocedor de señas. En paralelo, definir aquí el consentimiento informado y el
manejo de datos biométricos antes de subir nada a la nube.

> Nota sobre la emoción: el clasificador facial está planeado y su andamiaje existe,
> pero recomiendo tratarlo como una mejora posterior, no como parte de la primera
> versión usable. Hoy no tiene datos y mostrarlo daría una impresión equivocada.
