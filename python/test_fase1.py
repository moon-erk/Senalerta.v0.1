# -*- coding: utf-8 -*-
"""
test_fase1.py -- Prueba completa de los componentes de la Fase 1.
Ejecutar con: C:\v310\Scripts\python.exe python/test_fase1.py
"""

import sys, os, traceback
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

PASS = '[PASS]'
FAIL = '[FAIL]'
INFO = '[INFO]'

results = []

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    print(f'  {status}  {name}')
    if detail:
        print(f'         {detail}')
    results.append((name, condition))
    return condition

# ================================================================
print('\n=== PRUEBA 1 --- augmentation.py ===')
# ================================================================
try:
    from augmentation import augment_sequence, _LH_START, _RH_START, _LH_LEN, _RH_LEN

    # 1a. Forma de salida correcta
    dummy = [np.random.rand(1662).astype(np.float32) for _ in range(15)]
    augmented = augment_sequence(dummy, 5)
    check('Devuelve 5 variaciones', len(augmented) == 5,
          f'len={len(augmented)}')
    check('Cada variacion tiene 15 frames', all(len(v)==15 for v in augmented),
          f'frames={[len(v) for v in augmented]}')
    check('Cada frame tiene exactamente 1662 valores',
          all(len(f)==1662 for v in augmented for f in v),
          f'primer frame size={len(augmented[0][0])}')

    # 1b. Rotacion: las coordenadas x,y cambian, vis NO
    from augmentation import _rotate
    frame_rot = np.zeros(1662, dtype=np.float32)
    frame_rot[0] = 0.5   # pose landmark 0, x
    frame_rot[1] = 0.3   # pose landmark 0, y
    frame_rot[3] = 0.9   # pose landmark 0, visibilidad -- NO debe cambiar
    before_vis = frame_rot[3]
    rotated = _rotate(frame_rot.copy(), 15.0)
    check('Rotacion cambia x,y de un landmark',
          abs(rotated[0] - 0.5) > 1e-4 or abs(rotated[1] - 0.3) > 1e-4,
          f'x: {frame_rot[0]:.4f}->{rotated[0]:.4f}  y: {frame_rot[1]:.4f}->{rotated[1]:.4f}')
    check('Rotacion NO toca la visibilidad de pose',
          rotated[3] == before_vis,
          f'vis antes={before_vis:.4f}  despues={rotated[3]:.4f}')

    # 1c. Espejo: LH <-> RH, x invertida
    from augmentation import _flip_horizontal
    frame_flip = np.zeros(1662, dtype=np.float32)
    frame_flip[_LH_START]   = 0.1   # LH x[0]
    frame_flip[_LH_START+1] = 0.2   # LH y[0]
    frame_flip[_RH_START]   = 0.7   # RH x[0]
    frame_flip[_RH_START+1] = 0.8   # RH y[0]
    frame_flip[0] = 0.4              # Pose x
    flipped = _flip_horizontal(frame_flip.copy())
    check('Flip: LH pasa a posicion RH',
          abs(flipped[_RH_START] - 0.1) < 1e-4,
          f'LH_x original=0.1  -> RH_x despues={flipped[_RH_START]:.4f}')
    check('Flip: RH pasa a posicion LH',
          abs(flipped[_LH_START] - 0.7) < 1e-4,
          f'RH_x original=0.7  -> LH_x despues={flipped[_LH_START]:.4f}')
    check('Flip: x de pose se invierte (-x)',
          abs(flipped[0] - (-0.4)) < 1e-4,
          f'pose_x original=0.4  -> despues={flipped[0]:.4f}  esperado=-0.4')

    # 1d. Mano ausente (zeros) se restaura a cero tras el flip
    frame_absent = np.zeros(1662, dtype=np.float32)
    frame_absent[_RH_START]   = 0.5   # solo RH presente, LH ausente
    frame_absent[_RH_START+1] = 0.3
    flipped_absent = _flip_horizontal(frame_absent.copy())
    # Old LH (absent) moved to RH position after swap -- RH should stay zeros
    check('Flip: LH ausente -> posicion RH queda en zeros',
          np.all(flipped_absent[_RH_START: _RH_START + _RH_LEN] == 0),
          f'RH section all-zero={np.all(flipped_absent[_RH_START:_RH_START+_RH_LEN]==0)}')

    # 1e. Ruido: cambia valores pero dentro del rango esperado
    from augmentation import _noise
    frame_noise = np.ones(1662, dtype=np.float32) * 0.5
    noisy = _noise(frame_noise.copy(), std=0.01)
    delta = np.abs(noisy - frame_noise)
    check('Ruido gaussiano: desviacion media aprox 0.01',
          0.005 < delta.mean() < 0.02,
          f'desviacion media={delta.mean():.5f}  (esperado ~0.01)')

    # 1f. Augment con N=0 funciona
    aug_zero = augment_sequence(dummy, 0)
    check('augment_sequence(seq, 0) devuelve lista vacia', len(aug_zero) == 0)

except Exception as e:
    print(f'  {FAIL}  Excepcion en prueba de augmentation: {e}')
    traceback.print_exc()

# ================================================================
print('\n=== PRUEBA 2 --- helpers.py -> normalize_frame ===')
# ================================================================
try:
    from helpers import normalize_frame

    # 2a. Forma de salida
    frame = np.random.rand(1662).astype(np.float32)
    result = normalize_frame(frame)
    check('normalize_frame: salida tiene 1662 valores', len(result) == 1662,
          f'len={len(result)}')

    # 2b. Valores esperados con hombros conocidos
    frame_test = np.zeros(1662, dtype=np.float32)
    frame_test[44] = 0.4   # x hombro izq
    frame_test[45] = 0.5   # y hombro izq
    frame_test[46] = 0.0   # z hombro izq
    frame_test[48] = 0.6   # x hombro der
    frame_test[49] = 0.5   # y hombro der
    frame_test[50] = 0.0   # z hombro der
    # Centro=(0.5,0.5,0), distancia=0.2
    normed = normalize_frame(frame_test.copy())
    ls_x_norm = normed[44]
    check('normalize_frame: hombro izq normalizado correcto',
          abs(ls_x_norm - (-0.5)) < 1e-3,
          f'ls_x esperado=-0.5  obtenido={ls_x_norm:.4f}')
    rs_x_norm = normed[48]
    check('normalize_frame: hombro der normalizado correcto',
          abs(rs_x_norm - 0.5) < 1e-3,
          f'rs_x esperado=+0.5  obtenido={rs_x_norm:.4f}')

    # 2c. Sin hombros -> frame sin cambio
    frame_zero = np.zeros(1662, dtype=np.float32)
    frame_zero[100] = 0.7
    normed_zero = normalize_frame(frame_zero.copy())
    check('normalize_frame: sin hombros detectados -> frame sin cambio',
          abs(float(normed_zero[100]) - 0.7) < 1e-5,
          f'valor[100]={normed_zero[100]:.6f}  esperado=0.7')

    # 2d. Visibilidad de pose no se toca (indice 3 = visibilidad landmark 0)
    frame_vis = np.zeros(1662, dtype=np.float32)
    frame_vis[44] = 0.4; frame_vis[48] = 0.6
    frame_vis[3] = 0.99
    normed_vis = normalize_frame(frame_vis.copy())
    check('normalize_frame: visibilidad de pose no se modifica',
          abs(normed_vis[3] - 0.99) < 1e-4,
          f'vis antes=0.99  despues={normed_vis[3]:.4f}')

except Exception as e:
    print(f'  {FAIL}  Excepcion en prueba de normalize_frame: {e}')
    traceback.print_exc()

# ================================================================
print('\n=== PRUEBA 3 --- model.py -> arquitectura Bidirectional ===')
# ================================================================
try:
    from model import get_model

    m = get_model(15, 6)   # 15 frames, 6 senas de prueba

    # 3a. Capas esperadas
    layer_types = [type(l).__name__ for l in m.layers]
    check('Tiene capa Bidirectional',
          'Bidirectional' in layer_types,
          f'capas={layer_types}')
    check('Tiene BatchNormalization (>=2)',
          layer_types.count('BatchNormalization') >= 2,
          f'BatchNorm count={layer_types.count("BatchNormalization")}')
    check('Tiene Dropout (>=2)',
          layer_types.count('Dropout') >= 2,
          f'Dropout count={layer_types.count("Dropout")}')
    check('Capa de salida tiene 6 unidades (softmax)',
          m.layers[-1].units == 6,
          f'output units={m.layers[-1].units}')

    # 3b. Forward pass con datos sinteticos
    x_test = np.random.rand(4, 15, 1662).astype(np.float32)
    pred = m.predict(x_test, verbose=0)
    check('Forward pass OK: shape (4,6)',
          pred.shape == (4, 6),
          f'pred.shape={pred.shape}')
    check('Softmax: probabilidades suman ~1 por muestra',
          all(abs(row.sum() - 1.0) < 1e-5 for row in pred),
          f'sumas={[round(float(r.sum()),5) for r in pred]}')

    # 3c. Mini-entrenamiento 5 epocas
    from keras.utils import to_categorical
    X_syn = np.random.rand(30, 15, 1662).astype(np.float32)
    y_syn = to_categorical(np.random.randint(0, 6, 30), num_classes=6)
    hist = m.fit(X_syn, y_syn, epochs=5, batch_size=8, verbose=0)
    losses = hist.history['loss']
    check('Mini-entrenamiento 5 epocas sin error',
          len(losses) == 5,
          f'losses={[round(l,4) for l in losses]}')

except Exception as e:
    print(f'  {FAIL}  Excepcion en prueba de model: {e}')
    traceback.print_exc()

# ================================================================
print('\n=== PRUEBA 4 --- training_model.py con datos sinteticos ===')
# ================================================================
try:
    import pandas as pd
    import tempfile, json, shutil
    from augmentation import augment_sequence

    print(f'  {INFO}  Simulando pipeline completo de entrenamiento...')

    n_clases = 4
    n_muestras_por_clase = 8
    sequences_raw, labels_raw = [], []
    for cls in range(n_clases):
        for _ in range(n_muestras_por_clase):
            seq = []
            for _ in range(10):
                frame = np.random.rand(1662).astype(np.float32)
                frame[cls * 10: cls * 10 + 5] += 2.0
                frame[44] = 0.4; frame[45] = 0.5
                frame[48] = 0.6; frame[49] = 0.5
                seq.append(frame)
            sequences_raw.append(seq)
            labels_raw.append(cls)

    # 4a. Normalizacion
    from helpers import normalize_frame
    sequences_norm = [[normalize_frame(f) for f in seq] for seq in sequences_raw]
    check('normalize_frame aplicado a todas las secuencias',
          len(sequences_norm) == len(sequences_raw),
          f'{len(sequences_norm)} secuencias normalizadas')

    # 4b. Aumento x5
    aug_seqs, aug_labels = [], []
    for seq, label in zip(sequences_norm, labels_raw):
        variations = augment_sequence(seq, 5)
        aug_seqs.extend(variations)
        aug_labels.extend([label] * 5)
    total_seqs   = sequences_norm + aug_seqs
    total_labels = labels_raw + aug_labels
    check(f'Aumento x5: {len(sequences_raw)} -> {len(total_seqs)} muestras',
          len(total_seqs) == len(sequences_raw) * 6,
          f'original={len(sequences_raw)}  total={len(total_seqs)}')

    # 4c. Pad + split
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    padded = pad_sequences(total_seqs, maxlen=15, padding='pre',
                           truncating='post', dtype='float32')
    X = np.array(padded, dtype=np.float32)
    y = to_categorical(total_labels, num_classes=n_clases).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    check('pad_sequences OK: shape correcto',
          X.shape[1:] == (15, 1662),
          f'X.shape={X.shape}  (esperado Nx15x1662)')

    # 4d. Entrenamiento rapido (10 epocas)
    from model import get_model
    from tensorflow.keras.callbacks import EarlyStopping
    model_syn = get_model(15, n_clases)
    hist = model_syn.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=10, batch_size=8, verbose=0)
    check('Pipeline de entrenamiento completa sin errores',
          len(hist.history['accuracy']) > 0,
          f'accuracy epocas: {[round(a,3) for a in hist.history["accuracy"]]}')

    # 4e. Matriz de confusion
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = model_syn.predict(X_val, verbose=0)
    y_true_idx = np.argmax(y_val, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    tmp_path = os.path.join(tempfile.gettempdir(), 'test_confusion_matrix.png')
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=[f'sena_{i}' for i in range(n_clases)]).plot(ax=ax)
    plt.tight_layout()
    plt.savefig(tmp_path, dpi=100)
    plt.close(fig)
    check('Matriz de confusion generada y guardada',
          os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000,
          f'archivo={tmp_path}  size={os.path.getsize(tmp_path)} bytes')

    print(f'  {INFO}  Matriz de confusion (datos sinteticos):')
    print(f'         {cm}')

except Exception as e:
    print(f'  {FAIL}  Excepcion en prueba de pipeline: {e}')
    traceback.print_exc()

# ================================================================
print('\n=== PRUEBA 5 --- server.py importa normalize_frame ===')
# ================================================================
try:
    with open('server.py', encoding='utf-8') as f:
        src = f.read()

    check('server.py importa normalize_frame',
          'normalize_frame' in src)
    check('capture_loop llama normalize_frame antes de acumular keypoints',
          'normalize_frame(extract_keypoints(results))' in src)
    check('Endpoint /train usa augment_sequence',
          'augment_sequence' in src)
    check('Endpoint /train genera matriz de confusion',
          'confusion_matrix' in src)

except Exception as e:
    print(f'  {FAIL}  Excepcion en prueba de server: {e}')

# ================================================================
print('\n=== RESUMEN FINAL ===')
# ================================================================
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f'\n  Resultado: {passed}/{total} pruebas pasadas')
for name, ok in results:
    icon = 'OK' if ok else 'XX'
    print(f'    [{icon}]  {name}')
sys.exit(0 if passed == total else 1)
