"""
augmentation.py — Aumento de datos para secuencias de keypoints LSC.

Formato de entrada: lista de frames, cada frame es un vector de 1662 valores:
  [pose 33×4 (x,y,z,vis)] + [face 468×3 (x,y,z)] + [lh 21×3] + [rh 21×3]

Las transformaciones actúan solo sobre x,y,z (no sobre la visibilidad de pose).
"""

import numpy as np

# ── Distribución del vector de 1662 valores ───────────────────────────────────
_POSE_N  = 33
_FACE_N  = 468
_HAND_N  = 21

_POSE_LEN = _POSE_N * 4   # 132  — stride 4 (x,y,z,vis)
_FACE_LEN = _FACE_N * 3   # 1404 — stride 3 (x,y,z)
_LH_LEN   = _HAND_N * 3   # 63
_RH_LEN   = _HAND_N * 3   # 63

_FACE_START = _POSE_LEN                    # 132
_LH_START   = _FACE_START + _FACE_LEN     # 1536
_RH_START   = _LH_START + _LH_LEN         # 1599

# Índices de las coordenadas x, y, z dentro de los 1662 valores
_POSE_X = np.arange(0, _POSE_N * 4, 4)
_POSE_Y = _POSE_X + 1
_POSE_Z = _POSE_X + 2

_FACE_X = np.arange(_FACE_START, _FACE_START + _FACE_LEN, 3)
_FACE_Y = _FACE_X + 1
_FACE_Z = _FACE_X + 2

_LH_X = np.arange(_LH_START, _LH_START + _LH_LEN, 3)
_LH_Y = _LH_X + 1
_LH_Z = _LH_X + 2

_RH_X = np.arange(_RH_START, _RH_START + _RH_LEN, 3)
_RH_Y = _RH_X + 1
_RH_Z = _RH_X + 2

_ALL_X = np.concatenate([_POSE_X, _FACE_X, _LH_X, _RH_X])
_ALL_Y = np.concatenate([_POSE_Y, _FACE_Y, _LH_Y, _RH_Y])
_ALL_Z = np.concatenate([_POSE_Z, _FACE_Z, _LH_Z, _RH_Z])


# ── Transformaciones individuales ─────────────────────────────────────────────

def _rotate(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota todos los landmarks en el plano XY."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    xs = frame[_ALL_X].copy()
    ys = frame[_ALL_Y].copy()
    frame[_ALL_X] = c * xs - s * ys
    frame[_ALL_Y] = s * xs + c * ys
    return frame


def _translate(frame: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Desplaza todos los landmarks por un offset uniforme."""
    frame[_ALL_X] += dx
    frame[_ALL_Y] += dy
    frame[_ALL_Z] += dz
    return frame


def _scale(frame: np.ndarray, factor: float) -> np.ndarray:
    """Escala todos los landmarks alrededor del origen."""
    frame[_ALL_X] *= factor
    frame[_ALL_Y] *= factor
    frame[_ALL_Z] *= factor
    return frame


def _noise(frame: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Añade ruido gaussiano a todas las coordenadas x,y,z."""
    all_idx = np.concatenate([_ALL_X, _ALL_Y, _ALL_Z])
    frame[all_idx] += np.random.normal(0.0, std, len(all_idx)).astype(np.float32)
    return frame


def _flip_horizontal(frame: np.ndarray) -> np.ndarray:
    """
    Espejo horizontal: invierte la coordenada x (-x) y cambia LH ↔ RH.
    Usa -x porque el aumento se aplica DESPUÉS de normalize_frame, que centra
    el cuerpo en x=0 (con 1-x los datos quedarían desplazados un ancho de
    hombros completo, fuera de la distribución del tiempo real).
    Las secciones ausentes (todo cero) se restauran a cero tras el intercambio.
    """
    # Guarda máscaras de ausencia antes de modificar
    lh_data = frame[_LH_START: _LH_START + _LH_LEN].copy()
    rh_data = frame[_RH_START: _RH_START + _RH_LEN].copy()
    lh_absent = np.all(lh_data == 0)
    rh_absent = np.all(rh_data == 0)

    # Invierte x para todos los landmarks (incluso ausentes, lo corregimos después)
    frame[_ALL_X] = -frame[_ALL_X]

    # Intercambia LH ↔ RH
    frame[_LH_START: _LH_START + _LH_LEN] = rh_data
    frame[_RH_START: _RH_START + _RH_LEN] = lh_data

    # Restaura secciones que estaban ausentes
    if lh_absent:
        frame[_RH_START: _RH_START + _RH_LEN] = 0.0  # era LH ausente, ahora en posición RH
    if rh_absent:
        frame[_LH_START: _LH_START + _LH_LEN] = 0.0  # era RH ausente, ahora en posición LH

    return frame


# ── Función principal de aumento ──────────────────────────────────────────────

def augment_sequence(sequence, n: int) -> list:
    """
    Genera N variaciones aumentadas de una secuencia de keypoints.

    Args:
        sequence: lista de frames (cada frame es array/lista de 1662 valores).
        n:        número de variaciones a generar.

    Returns:
        Lista de n secuencias aumentadas, cada una con la misma longitud que
        la secuencia original y exactamente 1662 valores por frame.
    """
    all_transforms = ['rotate', 'translate', 'scale', 'noise', 'flip']
    augmented = []

    for _ in range(n):
        # Selecciona un subconjunto aleatorio de transformaciones para esta variación
        k = np.random.randint(1, len(all_transforms) + 1)
        chosen = set(np.random.choice(all_transforms, size=k, replace=False))

        # Parámetros fijos para toda la secuencia (coherencia temporal)
        angle  = float(np.random.uniform(-10, 10))         if 'rotate'    in chosen else 0.0
        dx     = float(np.random.uniform(-0.05, 0.05))     if 'translate' in chosen else 0.0
        dy     = float(np.random.uniform(-0.05, 0.05))     if 'translate' in chosen else 0.0
        dz     = float(np.random.uniform(-0.05, 0.05))     if 'translate' in chosen else 0.0
        factor = float(np.random.uniform(0.9, 1.1))        if 'scale'     in chosen else 1.0

        new_seq = []
        for raw_frame in sequence:
            frame = np.array(raw_frame, dtype=np.float32).copy()

            if 'rotate'    in chosen: frame = _rotate(frame, angle)
            if 'translate' in chosen: frame = _translate(frame, dx, dy, dz)
            if 'scale'     in chosen: frame = _scale(frame, factor)
            if 'noise'     in chosen: frame = _noise(frame)
            if 'flip'      in chosen: frame = _flip_horizontal(frame)

            new_seq.append(frame)

        augmented.append(new_seq)

    return augmented
