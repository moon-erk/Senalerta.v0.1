import os, sys, json, threading, base64, time
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from mediapipe.python.solutions.holistic import Holistic
from waitress import serve

sys.path.insert(0, os.path.dirname(__file__))
from constants import (APPDATA_PATH, FRAME_ACTIONS_PATH, KEYPOINTS_PATH,
                       MODEL_FOLDER_PATH, MODEL_PATH, WORDS_JSON_PATH,
                       MIN_LENGTH_FRAMES, MODEL_FRAMES, LENGTH_KEYPOINTS,
                       EMOTIONS_LIST, EMOTIONS_PATH,
                       EMOTION_MODEL_PATH, EMOTION_WORDS_PATH,
                       SIGNS_METADATA_PATH, LETTER_SAMPLE_FRAMES,
                       LETTER_STABLE_FRAMES)
from helpers import (mediapipe_detection, there_hand, extract_keypoints,
                     draw_keypoints, save_frames, get_word_ids,
                     get_keypoints, insert_keypoints_sequence,
                     normalize_keypoints, normalize_frame,
                     extract_face_keypoints, normalize_face_frame)
from model import get_model

app = Flask(__name__)
CORS(app, origins='*')

# ── ESTADO GLOBAL ──────────────────────────────────────────────────────────────
holistic_model  = None
trained_model   = None
cap              = None
camera_running   = False
latest_frame     = None
frame_lock       = threading.Lock()
capture_thread   = None
inference_thread = None

# evaluación en tiempo real
eval_kp_seq      = []
eval_sentence    = []
eval_count_frame = 0
eval_fix_frames  = 0
eval_recording   = False

# captura de muestras
sample_word             = None
sample_type             = 'word'    # 'word' (movimiento) o 'letter' (postura estática)
sample_frames           = []
sample_count_frame      = 0
sample_fix_frames       = 0
sample_recording        = False
sample_recording_active = False

# entrenamiento de señas
training_status = {'running': False, 'progress': 0, 'message': '', 'error': None}

# modelo y captura de emociones
emotion_model          = None
emotion_ids_loaded     = []
emotion_capture_active = False
emotion_current        = None
emotion_frame_counter  = 0
emotion_loop_counter   = 0
emotion_train_status   = {'running': False, 'progress': 0, 'message': '', 'error': None}

# ── INICIALIZACIÓN DE MEDIAPIPE ────────────────────────────────────────────────
def init_holistic():
    global holistic_model
    holistic_model = Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print('MediaPipe Holistic inicializado')

# ── CONFIANZA EN TIEMPO REAL ────────────────────────────────────────────────────
def get_realtime_confidence():
    if not trained_model or len(eval_kp_seq) < 3:
        return None, 0.0
    word_ids = get_word_ids()
    if not word_ids:
        return None, 0.0
    try:
        seq        = eval_kp_seq[-MODEL_FRAMES:] if len(eval_kp_seq) >= MODEL_FRAMES else eval_kp_seq
        normalized = normalize_keypoints(seq, MODEL_FRAMES)
        arr        = np.expand_dims(normalized, axis=0)
        res        = trained_model.predict(arr, verbose=0)[0]
        idx        = int(np.argmax(res))
        confidence = float(res[idx])
        if confidence > 0.3:
            return word_ids[idx], confidence
    except Exception:
        pass
    return None, 0.0

# ── SERIALIZAR LANDMARKS ────────────────────────────────────────────────────────
def encode_landmarks(results):
    def lm_to_list(lm_obj):
        if not lm_obj:
            return []
        return [{'id': i, 'x': round(lm.x, 4), 'y': round(lm.y, 4), 'z': round(lm.z, 4)}
                for i, lm in enumerate(lm_obj.landmark)]
    return {
        'left_hand':  lm_to_list(results.left_hand_landmarks),
        'right_hand': lm_to_list(results.right_hand_landmarks),
        'pose':       lm_to_list(results.pose_landmarks),
        'face':       lm_to_list(results.face_landmarks),
    }

# ── METADATOS DE SEÑAS (tipo letra/palabra) ────────────────────────────────────
def _load_signs_metadata():
    """Devuelve {seña: 'letter'|'word'}. Señas sin registro son 'word'."""
    if not os.path.exists(SIGNS_METADATA_PATH):
        return {}
    try:
        with open(SIGNS_METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_sign_type(word, sign_type):
    meta = _load_signs_metadata()
    meta[word] = sign_type
    with open(SIGNS_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ── CARGAR IDs DE EMOCIÓN ──────────────────────────────────────────────────────
def _load_emotion_ids():
    if not os.path.exists(EMOTION_WORDS_PATH):
        return []
    with open(EMOTION_WORDS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f).get('emotion_ids', [])

# ── GUARDAR MUESTRA DE EMOCIÓN ─────────────────────────────────────────────────
def _save_emotion_sample(emotion, face_kp):
    emotion_dir = os.path.join(EMOTIONS_PATH, emotion)
    os.makedirs(emotion_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
    filepath  = os.path.join(emotion_dir, f'sample_{timestamp}.json')
    with open(filepath, 'w') as f:
        json.dump(face_kp.tolist(), f)

# ── GUARDAR MUESTRA ─────────────────────────────────────────────────────────────
def _save_sample(word_id, frames):
    word_path   = os.path.join(FRAME_ACTIONS_PATH, word_id)
    os.makedirs(word_path, exist_ok=True)
    today       = datetime.now().strftime('%y%m%d%H%M%S%f')
    sample_path = os.path.join(word_path, f'sample_{today}')
    os.makedirs(sample_path, exist_ok=True)
    save_frames(frames, sample_path)
    print(f'Muestra guardada: {sample_path}')

# ── LOOPS DE CAPTURA E INFERENCIA (DOS HILOS) ──────────────────────────────────
# Diagnóstico 2026-06-11: MediaPipe (~290 ms) y el predict de confianza
# (~550 ms) bloqueaban el stream entero a 1-3 FPS. Desacoplados en dos hilos,
# el stream publica al ritmo de la cámara y el reconocimiento corre a su
# propio paso sobre el frame más reciente, sin frenar el video.

latest_raw    = None              # último frame crudo para el hilo de inferencia
raw_lock      = threading.Lock()
infer_results = {'landmarks': {'left_hand': [], 'right_hand': [], 'pose': [], 'face': []},
                 'has_hand': False, 'current_word': None, 'confidence': 0.0,
                 'emotion': None, 'emotion_confidence': 0.0}
infer_lock    = threading.Lock()

def capture_loop():
    """Hilo 1: lee la cámara, codifica JPEG (una sola vez) y publica el frame."""
    global latest_frame, latest_raw, camera_running
    _fps_times = []

    while camera_running:
        if cap is None or not cap.isOpened():
            break

        # lectura continua: con BUFFERSIZE=1 y el loop rápido no queda buffer
        # rancio, así que no hacen falta grabs extra (descartaban 2 de cada 3)
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        with raw_lock:
            latest_raw = frame

        # única codificación JPEG por frame
        _, buf  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_b64 = base64.b64encode(buf).decode('utf-8')

        _now_t = time.time()
        _fps_times.append(_now_t)
        _fps_times = [t for t in _fps_times if _now_t - t < 1.0]

        with infer_lock:
            res = dict(infer_results)
        with frame_lock:
            latest_frame = {
                'image':              img_b64,
                'landmarks':          res['landmarks'],
                'sentence':           list(eval_sentence),
                'has_hand':           res['has_hand'],
                'current_word':       res['current_word'],
                'confidence':         round(res['confidence'], 3),
                'emotion':            res['emotion'],
                'emotion_confidence': round(res['emotion_confidence'], 3),
                'fps':                len(_fps_times),
            }

        time.sleep(0.001)

def inference_loop():
    """Hilo 2: MediaPipe + reconocimiento sobre el frame más reciente."""
    global camera_running
    global eval_kp_seq, eval_sentence, eval_count_frame, eval_fix_frames, eval_recording
    global sample_frames, sample_count_frame, sample_fix_frames, sample_recording, sample_recording_active
    global emotion_capture_active, emotion_current, emotion_frame_counter, emotion_loop_counter

    MARGIN = 1
    DELAY  = 3
    conf_counter = 0
    last_word, last_conf = None, 0.0
    cur_emotion, cur_emotion_conf = None, 0.0
    last_frame_obj = None

    while camera_running:
        with raw_lock:
            frame = latest_raw
        if frame is None or frame is last_frame_obj:
            time.sleep(0.005)
            continue
        last_frame_obj = frame

        small   = cv2.resize(frame, (320, 240))
        results = mediapipe_detection(small, holistic_model)

        # ── EVALUACIÓN EN TIEMPO REAL ───────────────────────────────────────
        if there_hand(results) or eval_recording:
            eval_recording = False
            eval_count_frame += 1
            if eval_count_frame > MARGIN:
                # normalize_frame garantiza consistencia con el entrenamiento
                eval_kp_seq.append(normalize_frame(extract_keypoints(results)))
        else:
            if eval_count_frame >= MIN_LENGTH_FRAMES + MARGIN:
                eval_fix_frames += 1
                if eval_fix_frames < DELAY:
                    eval_recording = True
                else:
                    seq = eval_kp_seq[:-(MARGIN + DELAY)]
                    if trained_model and len(seq) >= MIN_LENGTH_FRAMES:
                        try:
                            normalized = normalize_keypoints(seq, MODEL_FRAMES)
                            res = trained_model.predict(
                                np.expand_dims(normalized, axis=0), verbose=0)[0]
                            if res[np.argmax(res)] > 0.8:
                                word_ids = get_word_ids()
                                word     = word_ids[int(np.argmax(res))]
                                eval_sentence.insert(0, word)
                                if len(eval_sentence) > 10:
                                    eval_sentence.pop()
                                threading.Thread(
                                    target=_tts_async, args=(word,), daemon=True
                                ).start()
                        except Exception as e:
                            print('Eval error:', e)
                    eval_kp_seq      = []
                    eval_count_frame = 0
                    eval_fix_frames  = 0
                    eval_recording   = False

        # ── CAPTURA DE MUESTRAS ─────────────────────────────────────────────
        if sample_recording_active and sample_word:
            if sample_type == 'letter':
                # LETRA: postura estática. Tras LETTER_STABLE_FRAMES de mano
                # estable se capturan LETTER_SAMPLE_FRAMES y la muestra se
                # cierra sola — sin esperar a que la mano salga del encuadre.
                # Si la mano se mantiene, siguen saliendo muestras en serie.
                if there_hand(results):
                    sample_count_frame += 1
                    if sample_count_frame > LETTER_STABLE_FRAMES:
                        sample_frames.append(frame.copy())
                    if len(sample_frames) >= LETTER_SAMPLE_FRAMES:
                        _save_sample(sample_word, sample_frames)
                        sample_frames      = []
                        sample_count_frame = 0
                else:
                    sample_frames      = []
                    sample_count_frame = 0
            else:
                # PALABRA: movimiento completo, captura mientras las manos
                # están en el encuadre (comportamiento original)
                if there_hand(results) or sample_recording:
                    sample_recording = False
                    sample_count_frame += 1
                    if sample_count_frame > MARGIN:
                        sample_frames.append(frame.copy())
                else:
                    if len(sample_frames) >= MIN_LENGTH_FRAMES + MARGIN:
                        sample_fix_frames += 1
                        if sample_fix_frames < DELAY:
                            sample_recording = True
                        else:
                            to_save = sample_frames[:-(MARGIN + DELAY)]
                            _save_sample(sample_word, to_save)
                            sample_frames      = []
                            sample_count_frame = 0
                            sample_fix_frames  = 0
                            sample_recording   = False

        # ── CAPTURA DE EMOCIONES ─────────────────────────────────────────────
        if emotion_capture_active and emotion_current:
            face_kp = extract_face_keypoints(results)
            if np.any(face_kp != 0):
                emotion_frame_counter += 1
                if emotion_frame_counter >= 30:
                    _save_emotion_sample(emotion_current,
                                        normalize_face_frame(face_kp))
                    emotion_frame_counter = 0

        # ── CONFIANZA EN TIEMPO REAL ────────────────────────────────────────
        # el predict es lo más caro del loop (~550 ms): corre solo cada 3
        # iteraciones y entre medias se reutiliza el último resultado
        conf_counter += 1
        if conf_counter % 3 == 0:
            last_word, last_conf = get_realtime_confidence()

        # Predicción de emoción (cada 5 iteraciones, MLP es liviano);
        # se conserva la última emoción entre predicciones
        emotion_loop_counter += 1
        if emotion_model and emotion_ids_loaded and emotion_loop_counter % 5 == 0:
            cur_emotion, cur_emotion_conf = None, 0.0
            face_kp = extract_face_keypoints(results)
            if np.any(face_kp != 0):
                try:
                    face_kp = normalize_face_frame(face_kp)
                    pred = emotion_model.predict(
                        np.expand_dims(face_kp, 0), verbose=0)[0]
                    idx  = int(np.argmax(pred))
                    if float(pred[idx]) > 0.5 and idx < len(emotion_ids_loaded):
                        cur_emotion      = emotion_ids_loaded[idx]
                        cur_emotion_conf = float(pred[idx])
                except Exception:
                    pass

        # ── PUBLICAR RESULTADOS DE INFERENCIA (serialización única) ─────────
        landmarks_json = encode_landmarks(results)
        with infer_lock:
            infer_results['landmarks']          = landmarks_json
            infer_results['has_hand']           = bool(there_hand(results))
            infer_results['current_word']       = last_word
            infer_results['confidence']         = last_conf
            infer_results['emotion']            = cur_emotion
            infer_results['emotion_confidence'] = cur_emotion_conf

def _tts_async(word):
    try:
        from text_to_speech import text_to_speech
        text_to_speech(word)
    except Exception as e:
        print('TTS error:', e)

# ── ENDPOINTS CÁMARA ───────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': trained_model is not None})

@app.route('/camera/start', methods=['POST'])
def camera_start():
    global cap, camera_running
    if camera_running:
        return jsonify({'status': 'already_running'})
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)   # el driver lo recorta a su máximo real
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(5):
        cap.read()
    if not cap.isOpened():
        return jsonify({'error': 'camera_not_found'}), 500
    camera_running = True
    global capture_thread, inference_thread
    capture_thread   = threading.Thread(target=capture_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    capture_thread.start()
    inference_thread.start()
    return jsonify({'status': 'started'})

@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    global cap, camera_running, latest_raw
    camera_running = False
    # esperar a que los hilos salgan del loop ANTES de liberar la cámara:
    # liberar cap mientras capture_loop está dentro de cap.read() puede
    # tumbar el proceso entero en código nativo de DSHOW
    for t in (capture_thread, inference_thread):
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
    if cap:
        cap.release()
        cap = None
    with raw_lock:
        latest_raw = None
    return jsonify({'status': 'stopped'})

@app.route('/camera/frame')
def camera_frame():
    with frame_lock:
        data = latest_frame
    if not data:
        return jsonify({'error': 'no_frame'}), 503
    return jsonify(data)

@app.route('/camera/stream')
def camera_stream():
    def gen():
        last_img = None
        while True:
            with frame_lock:
                data = latest_frame
            # solo emite frames nuevos (sin duplicar el último)
            if data and data['image'] is not last_img:
                last_img = data['image']
                raw = base64.b64decode(last_img)
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + raw + b'\r\n'
            time.sleep(0.005)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── ENDPOINTS CAPTURA DE MUESTRAS ──────────────────────────────────────────────
@app.route('/capture/start', methods=['POST'])
def capture_start():
    global sample_word, sample_type, sample_frames, sample_count_frame
    global sample_fix_frames, sample_recording, sample_recording_active
    body = request.get_json()
    word = body.get('word', '').strip().lower().replace(' ', '_')
    sign_type = body.get('type', 'word')
    if not word:
        return jsonify({'error': 'word_required'}), 400
    if sign_type not in ('letter', 'word'):
        return jsonify({'error': 'invalid_type', 'valid': ['letter', 'word']}), 400
    _save_sign_type(word, sign_type)
    sample_word             = word
    sample_type             = sign_type
    sample_frames           = []
    sample_count_frame      = 0
    sample_fix_frames       = 0
    sample_recording        = False
    sample_recording_active = True
    return jsonify({'status': 'capturing', 'word': word, 'type': sign_type})

@app.route('/capture/stop', methods=['POST'])
def capture_stop():
    global sample_recording_active
    sample_recording_active = False
    return jsonify({'status': 'stopped'})

@app.route('/capture/samples')
def capture_samples_list():
    meta   = _load_signs_metadata()
    result = {}
    if os.path.exists(FRAME_ACTIONS_PATH):
        for word in os.listdir(FRAME_ACTIONS_PATH):
            word_path = os.path.join(FRAME_ACTIONS_PATH, word)
            if os.path.isdir(word_path):
                result[word] = {
                    'count': len(os.listdir(word_path)),
                    'type':  meta.get(word, 'word'),
                }
    return jsonify(result)

@app.route('/capture/delete', methods=['POST'])
def capture_delete():
    import shutil
    body      = request.get_json()
    word      = body.get('word', '').strip().lower().replace(' ', '_')
    word_path = os.path.join(FRAME_ACTIONS_PATH, word)
    if os.path.exists(word_path):
        shutil.rmtree(word_path)
    meta = _load_signs_metadata()
    if word in meta:
        del meta[word]
        with open(SIGNS_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return jsonify({'status': 'deleted'})

# ── ENDPOINTS KEYPOINTS ────────────────────────────────────────────────────────
@app.route('/keypoints/create', methods=['POST'])
def keypoints_create():
    def _run():
        words = [w for w in os.listdir(FRAME_ACTIONS_PATH)
                 if os.path.isdir(os.path.join(FRAME_ACTIONS_PATH, w))]
        with Holistic(model_complexity=0, enable_segmentation=False) as h:
            for word_id in words:
                word_frames_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
                hdf_path         = os.path.join(KEYPOINTS_PATH, f'{word_id}.h5')
                data             = pd.DataFrame([])
                for n_sample, sample_name in enumerate(sorted(os.listdir(word_frames_path)), 1):
                    sample_path = os.path.join(word_frames_path, sample_name)
                    if not os.path.isdir(sample_path):
                        continue
                    kp_seq = get_keypoints(h, sample_path)
                    data   = insert_keypoints_sequence(data, n_sample, kp_seq)
                data.to_hdf(hdf_path, key='data', mode='w')
                print(f'Keypoints creados: {word_id}')
        with open(WORDS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({'word_ids': words}, f, ensure_ascii=False, indent=2)
        print('words.json actualizado')
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'status': 'processing'})

@app.route('/keypoints/status')
def keypoints_status():
    processed = [f.replace('.h5','') for f in os.listdir(KEYPOINTS_PATH) if f.endswith('.h5')]
    total     = [w for w in os.listdir(FRAME_ACTIONS_PATH)
                 if os.path.isdir(os.path.join(FRAME_ACTIONS_PATH, w))]
    return jsonify({'processed': processed, 'total': total, 'done': set(processed) >= set(total)})

# ── ENDPOINTS ENTRENAMIENTO ────────────────────────────────────────────────────
@app.route('/train', methods=['POST'])
def train():
    global training_status
    if training_status['running']:
        return jsonify({'error': 'already_training'}), 409

    def _train():
        global trained_model, training_status
        try:
            from tensorflow.keras.callbacks import EarlyStopping, Callback
            from keras.utils import to_categorical
            from sklearn.model_selection import train_test_split

            training_status = {'running': True, 'progress': 10,
                               'message': 'Cargando datos...', 'error': None}
            word_ids = get_word_ids()
            if not word_ids:
                raise ValueError('No hay palabras. Crea los keypoints primero.')

            sequences, labels = [], []
            for wi, word_id in enumerate(word_ids):
                hdf_path = os.path.join(KEYPOINTS_PATH, f'{word_id}.h5')
                if not os.path.exists(hdf_path):
                    continue
                data = pd.read_hdf(hdf_path, key='data')
                for _, df_sample in data.groupby('sample'):
                    seq = [row['keypoints'] for _, row in df_sample.iterrows()]
                    sequences.append(seq)
                    labels.append(wi)

            training_status.update({'progress': 25, 'message': 'Normalizando frames...'})
            # Normalización corporal — consistente con evaluación en tiempo real
            sequences = [[normalize_frame(f) for f in seq] for seq in sequences]

            # Split ANTES del aumento: la validación contiene solo muestras
            # reales nunca vistas (sin variaciones de muestras de train)
            train_seqs, val_seqs, train_labels, val_labels = train_test_split(
                sequences, labels, test_size=0.1, random_state=42, stratify=labels
            )

            training_status.update({'progress': 30, 'message': 'Aumentando datos...'})
            # Aumento de datos × 5, SOLO al set de entrenamiento
            from augmentation import augment_sequence
            aug_seqs, aug_labels = [], []
            for seq, label in zip(train_seqs, train_labels):
                variations = augment_sequence(seq, 5)
                aug_seqs.extend(variations)
                aug_labels.extend([label] * 5)
            train_seqs   = train_seqs + aug_seqs
            train_labels = train_labels + aug_labels

            training_status.update({'progress': 40, 'message': 'Preparando tensores...'})
            # Ajuste temporal con normalize_keypoints (interpolación), el MISMO
            # método del loop en tiempo real — no pad/truncate
            def _to_tensors(seqs, lbls):
                resampled = [normalize_keypoints(list(s), int(MODEL_FRAMES)) for s in seqs]
                return (np.array(resampled, dtype=np.float32),
                        to_categorical(lbls, num_classes=len(word_ids)).astype(int))
            X_train, y_train = _to_tensors(train_seqs, train_labels)
            X_val,   y_val   = _to_tensors(val_seqs, val_labels)

            training_status.update({'progress': 50, 'message': 'Construyendo modelo...'})
            model = get_model(int(MODEL_FRAMES), len(word_ids))

            class ProgressCB(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pct = 55 + int((epoch / 500) * 40)
                    training_status['progress'] = min(pct, 95)
                    val_acc = logs.get('val_accuracy', logs.get('accuracy', 0))
                    training_status['message'] = (
                        f'Época {epoch+1} — acc: {logs.get("accuracy", 0):.3f}'
                        f' | val_acc: {val_acc:.3f}'
                    )

            training_status.update({'progress': 55, 'message': 'Entrenando...'})
            # Pesos de clase: compensa señas con muchas vs pocas muestras
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(train_labels)
            cw = compute_class_weight('balanced', classes=classes, y=train_labels)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=500, batch_size=16, class_weight=class_weight,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=20,
                                               restore_best_weights=True,
                                               start_from_epoch=15), ProgressCB()],
                      verbose=0)
            model.save(MODEL_PATH)
            trained_model = model

            # precisión final del modelo guardado (sobre muestras reales de val)
            _, final_val_acc   = model.evaluate(X_val, y_val, verbose=0)
            _, final_train_acc = model.evaluate(X_train, y_train, verbose=0)

            # Matriz de confusión
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                y_pred     = model.predict(X_val, verbose=0)
                y_true_idx = np.argmax(y_val, axis=1)
                y_pred_idx = np.argmax(y_pred, axis=1)
                cm = confusion_matrix(y_true_idx, y_pred_idx,
                                      labels=list(range(len(word_ids))))
                fig, ax = plt.subplots(figsize=(max(6, len(word_ids)), max(5, len(word_ids))))
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=word_ids).plot(
                    ax=ax, xticks_rotation=45, colorbar=False)
                ax.set_title('Matriz de confusión — Señas LSC')
                plt.tight_layout()
                cm_path = os.path.join(MODEL_FOLDER_PATH, 'confusion_matrix.png')
                plt.savefig(cm_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print('Matriz de confusión guardada:', cm_path)
            except Exception as cm_err:
                print('Advertencia: no se pudo guardar matriz de confusión:', cm_err)

            training_status = {'running': False, 'progress': 100,
                               'message': 'Entrenamiento completo', 'error': None,
                               'train_acc': round(float(final_train_acc), 4),
                               'val_acc':   round(float(final_val_acc), 4)}
            print('Modelo guardado:', MODEL_PATH)
        except Exception as e:
            training_status = {'running': False, 'progress': 0,
                               'message': '', 'error': str(e)}
            print('Error entrenamiento:', e)

    threading.Thread(target=_train, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/train/status')
def train_status():
    return jsonify(training_status)

# ── ENDPOINTS DE LECTURA DEL MODELO (calidad, matriz, estado) ──────────────────
# Todos son de SOLO LECTURA: no entrenan ni modifican el modelo. /model/quality
# evalúa el modelo guardado sobre las muestras reales para encontrar señas
# problemáticas (no es una métrica de validación limpia: sirve de control de
# calidad para saber qué señas reforzar).

@app.route('/model/confusion_matrix')
def model_confusion_matrix():
    cm_path = os.path.join(MODEL_FOLDER_PATH, 'confusion_matrix.png')
    if not os.path.exists(cm_path):
        return jsonify({'error': 'no_matrix',
                        'message': 'Entrena el modelo primero.'}), 404
    return send_file(cm_path, mimetype='image/png')

def _build_quality_report():
    """Evalúa el modelo guardado sobre las muestras reales de cada seña.
    Devuelve por seña: tipo, nº de muestras, precisión propia y confusiones."""
    meta     = _load_signs_metadata()
    word_ids = get_word_ids()
    report   = {'signs': [], 'trained': False, 'word_ids': word_ids}

    # conteo de muestras (frames grabados) por seña
    counts = {}
    if os.path.exists(FRAME_ACTIONS_PATH):
        for w in os.listdir(FRAME_ACTIONS_PATH):
            wp = os.path.join(FRAME_ACTIONS_PATH, w)
            if os.path.isdir(wp):
                counts[w] = len(os.listdir(wp))

    model = trained_model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            from keras.models import load_model
            model = load_model(MODEL_PATH)
        except Exception:
            model = None

    # matriz de confusión en memoria: filas = seña real, cols = predicha
    n = len(word_ids)
    cmatrix = [[0] * n for _ in range(n)]
    have_preds = False

    if model is not None and n > 0:
        for wi, word_id in enumerate(word_ids):
            hdf_path = os.path.join(KEYPOINTS_PATH, f'{word_id}.h5')
            if not os.path.exists(hdf_path):
                continue
            try:
                data = pd.read_hdf(hdf_path, key='data')
            except Exception:
                continue
            for _, df_sample in data.groupby('sample'):
                seq = [normalize_frame(r['keypoints']) for _, r in df_sample.iterrows()]
                if not seq:
                    continue
                arr = np.expand_dims(
                    np.array(normalize_keypoints(list(seq), int(MODEL_FRAMES)),
                             dtype=np.float32), axis=0)
                pred = int(np.argmax(model.predict(arr, verbose=0)[0]))
                cmatrix[wi][pred] += 1
                have_preds = True

    report['trained'] = have_preds

    for wi, word_id in enumerate(word_ids):
        total_pred = sum(cmatrix[wi]) if have_preds else 0
        correct    = cmatrix[wi][wi] if have_preds else 0
        acc        = (correct / total_pred) if total_pred else None
        # confusiones: otras señas a las que se predijo esta seña
        confusions = []
        if have_preds:
            for wj in range(len(word_ids)):
                if wj != wi and cmatrix[wi][wj] > 0:
                    confusions.append({'with': word_ids[wj], 'count': cmatrix[wi][wj]})
            confusions.sort(key=lambda c: -c['count'])
        n_samples = counts.get(word_id, 0)
        problematic = bool(
            (acc is not None and acc < 0.8) or
            n_samples < 20 or
            len(confusions) > 0
        )
        report['signs'].append({
            'name':        word_id,
            'type':        meta.get(word_id, 'word'),
            'samples':     n_samples,
            'accuracy':    round(acc, 3) if acc is not None else None,
            'confusions':  confusions,
            'problematic': problematic,
        })

    # señas con muestras pero aún no entrenadas (no están en word_ids)
    for w, c in counts.items():
        if w not in word_ids:
            report['signs'].append({
                'name': w, 'type': meta.get(w, 'word'), 'samples': c,
                'accuracy': None, 'confusions': [],
                'problematic': c < 20, 'untrained': True,
            })
    return report

@app.route('/model/quality')
def model_quality():
    try:
        return jsonify(_build_quality_report())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/status')
def model_status():
    meta     = _load_signs_metadata()
    word_ids = get_word_ids()
    counts   = {}
    if os.path.exists(FRAME_ACTIONS_PATH):
        for w in os.listdir(FRAME_ACTIONS_PATH):
            wp = os.path.join(FRAME_ACTIONS_PATH, w)
            if os.path.isdir(wp):
                counts[w] = len(os.listdir(wp))

    # tipos contados sobre las señas que el modelo conoce
    n_letters = sum(1 for w in word_ids if meta.get(w, 'word') == 'letter')
    n_words   = sum(1 for w in word_ids if meta.get(w, 'word') == 'word')

    last_trained = None
    if os.path.exists(MODEL_PATH):
        last_trained = datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M')

    return jsonify({
        'model_exists':   os.path.exists(MODEL_PATH),
        'known_signs':    len(word_ids),
        'word_ids':       word_ids,
        'letters':        n_letters,
        'words':          n_words,
        'total_samples':  sum(counts.get(w, 0) for w in word_ids),
        'all_samples':    sum(counts.values()),
        'last_trained':   last_trained,
        'last_val_acc':   training_status.get('val_acc'),
        'last_train_acc': training_status.get('train_acc'),
    })

@app.route('/model/sign_type', methods=['POST'])
def model_set_sign_type():
    """Persiste el tipo (letter/word) de una seña sin capturar — para el toggle
    'tiene movimiento' del modo serie."""
    body = request.get_json() or {}
    word = body.get('word', '').strip().lower().replace(' ', '_')
    sign_type = body.get('type', 'word')
    if not word:
        return jsonify({'error': 'word_required'}), 400
    if sign_type not in ('letter', 'word'):
        return jsonify({'error': 'invalid_type'}), 400
    _save_sign_type(word, sign_type)
    return jsonify({'status': 'ok', 'word': word, 'type': sign_type})

# ── ENDPOINTS EMOCIÓN ─────────────────────────────────────────────────────────
@app.route('/emotion/capture/start', methods=['POST'])
def emotion_capture_start():
    global emotion_capture_active, emotion_current, emotion_frame_counter
    body    = request.get_json()
    emotion = body.get('emotion', '').strip().lower()
    if emotion not in EMOTIONS_LIST:
        return jsonify({'error': f'Emocion invalida. Validas: {EMOTIONS_LIST}'}), 400
    emotion_current        = emotion
    emotion_frame_counter  = 0
    emotion_capture_active = True
    return jsonify({'status': 'capturing', 'emotion': emotion})

@app.route('/emotion/capture/stop', methods=['POST'])
def emotion_capture_stop():
    global emotion_capture_active, emotion_frame_counter
    emotion_capture_active = False
    emotion_frame_counter  = 0
    return jsonify({'status': 'stopped'})

@app.route('/emotion/samples')
def emotion_samples():
    result = {}
    for emotion in EMOTIONS_LIST:
        emotion_dir = os.path.join(EMOTIONS_PATH, emotion)
        if os.path.exists(emotion_dir):
            result[emotion] = len([f for f in os.listdir(emotion_dir)
                                   if f.endswith('.json')])
        else:
            result[emotion] = 0
    return jsonify(result)

@app.route('/emotion/train', methods=['POST'])
def emotion_train():
    global emotion_train_status
    if emotion_train_status['running']:
        return jsonify({'error': 'already_training'}), 409

    def _train_emotion():
        global emotion_model, emotion_ids_loaded, emotion_train_status
        try:
            from keras.utils import to_categorical
            from sklearn.model_selection import train_test_split
            from emotion_model import get_emotion_model

            emotion_train_status = {'running': True, 'progress': 10,
                                    'message': 'Cargando muestras...', 'error': None}

            X, y, emotion_ids = [], [], []
            for emotion in EMOTIONS_LIST:
                emotion_dir = os.path.join(EMOTIONS_PATH, emotion)
                if not os.path.exists(emotion_dir):
                    continue
                samples = [f for f in os.listdir(emotion_dir) if f.endswith('.json')]
                if not samples:
                    continue
                class_idx = len(emotion_ids)
                emotion_ids.append(emotion)
                for fname in samples:
                    with open(os.path.join(emotion_dir, fname)) as f:
                        kp = json.load(f)
                    X.append(kp)
                    y.append(class_idx)

            if len(emotion_ids) < 2:
                raise ValueError('Se necesitan muestras de al menos 2 emociones para entrenar.')

            emotion_train_status.update({'progress': 30,
                                         'message': f'Entrenando con {len(emotion_ids)} emociones...'})

            X_arr = np.array(X, dtype=np.float32)
            y_arr = to_categorical(y, num_classes=len(emotion_ids)).astype(int)
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr, y_arr, test_size=0.15, random_state=42
            )

            from tensorflow.keras.callbacks import EarlyStopping
            model = get_emotion_model(len(emotion_ids))
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=16,
                callbacks=[EarlyStopping(monitor='val_accuracy', patience=15,
                                         restore_best_weights=True)],
                verbose=0,
            )

            emotion_train_status.update({'progress': 85, 'message': 'Guardando modelo...'})
            model.save(EMOTION_MODEL_PATH)
            with open(EMOTION_WORDS_PATH, 'w', encoding='utf-8') as f:
                json.dump({'emotion_ids': emotion_ids}, f, ensure_ascii=False, indent=2)

            emotion_model      = model
            emotion_ids_loaded = emotion_ids

            val_acc = float(max(model.evaluate(X_val, y_val, verbose=0)[1], 0))
            emotion_train_status = {
                'running': False, 'progress': 100,
                'message': f'Completo — {len(emotion_ids)} emociones | val_acc={val_acc:.2%}',
                'error': None,
            }
            print(f'Modelo de emocion guardado: {EMOTION_MODEL_PATH}')

        except Exception as e:
            emotion_train_status = {'running': False, 'progress': 0,
                                    'message': '', 'error': str(e)}
            print('Error entrenamiento emocion:', e)

    threading.Thread(target=_train_emotion, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/emotion/train/status')
def emotion_train_status_endpoint():
    return jsonify(emotion_train_status)

# ── ENDPOINTS RECONOCIMIENTO ───────────────────────────────────────────────────
@app.route('/recognition/sentence')
def recognition_sentence():
    return jsonify({'sentence': list(eval_sentence)})

@app.route('/recognition/clear', methods=['POST'])
def recognition_clear():
    global eval_sentence
    eval_sentence = []
    return jsonify({'status': 'cleared'})

@app.route('/recognition/words')
def recognition_words():
    return jsonify({'word_ids': get_word_ids()})

# ── ARRANQUE ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Iniciando SeñAlerta backend...')
    init_holistic()
    if os.path.exists(MODEL_PATH):
        try:
            from keras.models import load_model
            trained_model = load_model(MODEL_PATH)
            # warmup: el primer predict traza el grafo de TF (>10 s); mejor
            # pagarlo aquí que al hacer la primera seña en vivo
            trained_model.predict(
                np.zeros((1, int(MODEL_FRAMES), LENGTH_KEYPOINTS), dtype=np.float32),
                verbose=0)
            print('Modelo de señas cargado:', MODEL_PATH)
        except Exception as e:
            print('No se pudo cargar modelo de señas:', e)
    if os.path.exists(EMOTION_MODEL_PATH):
        try:
            from keras.models import load_model as _lm
            emotion_model      = _lm(EMOTION_MODEL_PATH)
            emotion_ids_loaded = _load_emotion_ids()
            print(f'Modelo de emocion cargado: {EMOTION_MODEL_PATH} ({emotion_ids_loaded})')
        except Exception as e:
            print('No se pudo cargar modelo de emocion:', e)
    print('Backend listo en http://localhost:8765')
    serve(app, host='0.0.0.0', port=8765, threads=8)
