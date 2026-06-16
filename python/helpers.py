import json
import os
import cv2
import numpy as np
import pandas as pd
from typing import NamedTuple
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from constants import KEYPOINTS_PATH, WORDS_JSON_PATH

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results

def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks

def get_word_ids(path=WORDS_JSON_PATH):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f).get('word_ids', [])

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def draw_keypoints(image, results):
    draw_landmarks(image, results.face_landmarks, FACEMESH_CONTOURS,
        DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    draw_landmarks(image, results.pose_landmarks, POSE_CONNECTIONS,
        DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    draw_landmarks(image, results.left_hand_landmarks, HAND_CONNECTIONS,
        DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    draw_landmarks(image, results.right_hand_landmarks, HAND_CONNECTIONS,
        DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def save_frames(frames, output_folder):
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_folder, f'{i+1}.jpg'),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

def extract_keypoints(results):
    pose = np.array([[r.x,r.y,r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[r.x,r.y,r.z] for r in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468*3)
    lh   = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def extract_face_keypoints(results):
    """Extrae 468×3=1404 valores de face landmarks, o zeros si no hay cara."""
    if results.face_landmarks:
        return np.array([[r.x, r.y, r.z]
                         for r in results.face_landmarks.landmark]).flatten()
    return np.zeros(468 * 3)

def normalize_face_frame(face_kp):
    """
    Normaliza 1404 valores de face landmarks relativo a la cara.
    Origen: centroide de todos los landmarks.
    Escala: desviacion estandar de las coordenadas x.
    Hace el modelo invariante a posicion y distancia de la camara.
    """
    face_kp = np.array(face_kp, dtype=np.float32)
    xs = face_kp[0::3]
    ys = face_kp[1::3]
    zs = face_kp[2::3]
    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())
    scale = float(xs.std())
    if scale < 1e-6:
        return face_kp
    face_kp[0::3] = (xs - cx) / scale
    face_kp[1::3] = (ys - cy) / scale
    face_kp[2::3] = (zs - cz) / scale
    return face_kp

def get_keypoints(model, sample_path):
    kp_seq = np.array([])
    for img_name in sorted(os.listdir(sample_path)):
        img_path = os.path.join(sample_path, img_name)
        frame    = cv2.imread(img_path)
        if frame is None:
            continue
        results  = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq   = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq

def insert_keypoints_sequence(df, n_sample, kp_seq):
    for frame, keypoints in enumerate(kp_seq):
        row = pd.DataFrame({'sample': n_sample, 'frame': frame+1, 'keypoints': [keypoints]})
        df  = pd.concat([df, row])
    return df

def get_sequences_and_labels(word_ids):
    sequences, labels = [], []
    for word_index, word_id in enumerate(word_ids):
        hdf_path = os.path.join(KEYPOINTS_PATH, f'{word_id}.h5')
        if not os.path.exists(hdf_path):
            continue
        data = pd.read_hdf(hdf_path, key='data')
        for _, df_sample in data.groupby('sample'):
            seq = [row['keypoints'] for _, row in df_sample.iterrows()]
            sequences.append(seq)
            labels.append(word_index)
    return sequences, labels

def normalize_frame(frame):
    """
    Normaliza un frame de 1662 valores relativo al cuerpo.

    Origen:  punto medio entre los hombros (landmarks de pose 11 y 12).
    Escala:  distancia euclidiana entre ambos hombros.

    Si los hombros no están detectados (todos cero), devuelve el frame sin cambios.
    Aplica a todos los landmarks x,y,z (la visibilidad de pose no se toca).
    """
    frame = np.array(frame, dtype=np.float32)

    # Pose landmark 11 (hombro izq): índices 44,45,46  (11*4=44)
    # Pose landmark 12 (hombro der): índices 48,49,50  (12*4=48)
    ls_x, ls_y, ls_z = frame[44], frame[45], frame[46]
    rs_x, rs_y, rs_z = frame[48], frame[49], frame[50]

    # Si ambos hombros son cero, no hay referencia — devuelve sin cambio
    if ls_x == 0 and ls_y == 0 and rs_x == 0 and rs_y == 0:
        return frame

    cx = (ls_x + rs_x) / 2.0
    cy = (ls_y + rs_y) / 2.0
    cz = (ls_z + rs_z) / 2.0

    dist = float(np.sqrt((rs_x - ls_x)**2 + (rs_y - ls_y)**2 + (rs_z - ls_z)**2))
    if dist < 1e-6:
        return frame

    # Índices de todas las coordenadas x, y, z en el vector de 1662 valores
    pose_x = np.arange(0,    33 * 4,   4)
    face_x = np.arange(132,  132 + 468 * 3, 3)
    lh_x   = np.arange(1536, 1536 + 21 * 3, 3)
    rh_x   = np.arange(1599, 1599 + 21 * 3, 3)

    all_x = np.concatenate([pose_x, face_x, lh_x, rh_x])
    all_y = all_x + 1
    all_z = all_x + 2

    # Corrige poses: stride 4 → los +1, +2 son los correctos para pose también
    # Para face/lh/rh el stride es 3, así que +1 y +2 son y y z correctos

    # Normalización: centrar en origen del cuerpo y escalar por distancia entre hombros
    frame[all_x] = (frame[all_x] - cx) / dist
    frame[all_y] = (frame[all_y] - cy) / dist
    frame[all_z] = (frame[all_z] - cz) / dist

    return frame


def normalize_keypoints(keypoints, target=15):
    n = len(keypoints)
    if n == target:
        return keypoints
    if n < target:
        indices = np.linspace(0, n-1, target)
        result  = []
        for i in indices:
            lo, hi = int(np.floor(i)), int(np.ceil(i))
            w = i - lo
            if lo == hi:
                result.append(keypoints[lo])
            else:
                interp = (1-w)*np.array(keypoints[lo]) + w*np.array(keypoints[hi])
                result.append(interp.tolist())
        return result
    step    = n / target
    indices = np.arange(0, n, step).astype(int)[:target]
    return [keypoints[i] for i in indices]
