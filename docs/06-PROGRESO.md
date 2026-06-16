# Progreso de SeñAlerta

## Hecho
- Backend de reconocimiento de señas (funciona y probado)
- test_popup.html con diseño estilo Echo's Glasses
- Entorno Python 3.10.11 en C:\v310\
- Documento de arquitectura y documento maestro
- Decisiones de backend, frontend y UI/UX tomadas
- Paso A2 — Documentación de la red neuronal en /docs/red-neuronal/ (que-es-el-lstm, flujo-de-datos, arquitectura-del-modelo, bitacora-experimentos) — 2026-06-10
- Paso B1 — Aumento de datos: augmentation.py verificado (rotación, desplazamiento, escalado, ruido, espejo), bug del espejo corregido (1-x → -x para datos normalizados), training_model.py aplica ×5, reentrenado y comparado (ver bitácora) — 2026-06-11
- Fuga de datos corregida en training_model.py Y en el endpoint /train de server.py (mismo defecto duplicado): split train/val antes del aumento (estratificado), aumento solo a train; la precisión de validación ahora se mide sobre muestras reales nunca vistas — 2026-06-11
- Paso B2 — Normalización corporal verificada en TODO el pipeline: normalize_frame existe (hombros 11/12 como origen, distancia como escala) y se aplica consistentemente. Corregido: evaluate_model.py evaluaba con keypoints crudos; el entrenamiento usaba pad/truncate mientras el vivo interpola (ahora ambos usan normalize_keypoints); EarlyStopping ahora monitorea val_loss con start_from_epoch (val_accuracy restauraba modelos sin entrenar); class_weight balanceado. hola.h5 estaba desactualizado (2 de 96 muestras) y se regeneró. Prueba en vivo del pipeline: 5/6 con confianza >0.8; backend arrancado y verificado (/health OK, cámara 8 fps) — 2026-06-11

- Optimización de FPS del backend: loop de captura desacoplado en dos hilos (stream a ritmo de cámara + inferencia a su paso), predict cacheado cada 3 iteraciones, JPEG q60, stream sin duplicados, warmup del modelo al arrancar. Stream: 3 → 24.6 FPS reales (30-31 internos, máximo físico de la cámara). Reconocimiento en vivo verificado ("gracias" ×5 con conf >0.8) y precisión simulada intacta (5/6) — 2026-06-11

- test_popup.html diagnosticado y arreglado: el "no funciona" era el backend caído (proceso murió sin traceback, posiblemente por la carrera de camera_stop liberando la cámara mientras el hilo de captura leía — camera_stop ahora espera a que los hilos terminen antes de liberar). Bug adicional encontrado y corregido en el popup: id malformado `id="hud-tr err"` rompía checkHealth y el indicador quedaba en "Sin conexión" siempre. Verificado con Playwright: indicador verde, stream renderiza (640px, ~30fps), barra de confianza se mueve, endpoints de captura/entrenamiento responden, ciclo apagar/encender sobrevive — 2026-06-11

- Fase C (parte técnica) — Distinción LETRA/PALABRA + modo grabación en serie: /capture/start acepta type (letter/word) y lo guarda en models/signs_metadata.json; captura de LETRA se cierra sola tras 7 frames con la mano estable (sin sacarla del encuadre, muestras en serie) mientras PALABRA mantiene el comportamiento original; /capture/samples devuelve {count, type}; /train/status expone train_acc y val_acc al terminar. Popup: selector LETRA/PALABRA con explicación en la captura manual, y modo "Grabación en serie" (overlay con las 27 letras del abecedario, meta 20 muestras, contador en vivo, celebración visual, siguiente pendiente, pausar, barra global X/27, botón "Entrenar modelo base" habilitado solo con las 27 completas que ejecuta keypoints→train con barras de progreso y muestra la precisión). El modelo base de fábrica = SOLO abecedario; saludos/palabras los añade cada usuario. Verificado con Playwright (27 letras, type letter en la petición, botón entrenar deshabilitado) y el cierre automático de letra probado con el inference_loop real alimentado con frames grabados: 7 muestras de 7 frames en serie. Falta: grabar las 27 letras frente a la cámara (trabajo humano) y entrenar el modelo base — 2026-06-11

- Demo lista para mostrar: creado INICIAR_DEMO.bat (arranca backend, espera /health, abre el popup — un solo clic). Corregida la causa probable de las caídas silenciosas del backend: text_to_speech.py hacía pygame init/quit concurrente desde varios hilos con un archivo speech.mp3 compartido (crash nativo al reconocer señas seguidas); ahora el TTS está serializado con lock, mixer único y archivo temporal propio por reproducción — 2026-06-12

- Control de calidad del modelo base en el popup (5 funciones, HTML puro): (1) botón "Ver matriz de confusión" con visor modal servido por GET /model/confusion_matrix; (2) tabla "Calidad por seña" tras entrenar (tipo, muestras, precisión propia, con qué se confunde; problemáticas en rojo) desde GET /model/quality; (3) sección "Reforzar" en modo serie que lista las señas confundidas con botón "+10 muestras" sin reiniciar las existentes y opción de reentrenar; (4) toggle "tiene movimiento" por letra (J/Z/Ñ activadas por defecto) que captura como type word y persiste vía POST /model/sign_type; (5) panel "Estado del modelo" con datos reales (GET /model/status: nº señas, tipos, muestras, fecha de último entrenamiento, última precisión val.). Endpoints nuevos SOLO de lectura — no tocan el pipeline de entrenamiento. Verificado con Playwright: matriz 200 image/png + visor (707px), tabla con 2 filas en rojo, refuerzo con 2 filas y botones, toggle envía type correcto (arreglado doble-disparo de label→span), panel de estado con datos reales. Sin errores de página — 2026-06-13

## En proceso
- Potenciar el LSTM + clasificador de emoción (Claude Code trabajando)

## Siguiente
- Frontend nuevo y ordenado con las 7 pantallas
- Diseñar pantallas en Stitch
- Integrar Supabase (cuentas, QR, estadísticas)

## Notas
- (aquí anotas lo que vayas necesitando recordar)