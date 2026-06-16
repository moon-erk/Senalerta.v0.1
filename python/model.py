"""
model.py — Arquitectura del modelo LSTM de señas.

Arquitectura actual (v2 — Bidirectional + BatchNorm):
  Bidirectional(LSTM 64) → BatchNorm → Dropout(0.5)
  → LSTM(128) → BatchNorm → Dropout(0.5)
  → Dense(64, relu) → Dense(N, softmax)

Arquitectura original (v1 — para revertir si es necesario):
  LSTM(64, L2) → Dropout(0.5) → LSTM(128, L2) → Dropout(0.5)
  → Dense(64, relu) → Dense(64, relu) → Dense(N, softmax)
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS


def get_model(max_length_frames, output_length):
    """
    Construye el modelo LSTM mejorado para reconocimiento de señas.

    Args:
        max_length_frames: número de frames por secuencia (MODEL_FRAMES = 15)
        output_length:     número de clases (señas distintas)

    Returns:
        Modelo Keras compilado.
    """
    model = Sequential([
        # Capa 1: Bidirectional LSTM — captura contexto hacia adelante y atrás en la seña
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(max_length_frames, LENGTH_KEYPOINTS)
        ),
        BatchNormalization(),
        Dropout(0.5),

        # Capa 2: LSTM unidireccional — extrae representación final de la secuencia
        LSTM(128, return_sequences=False),
        BatchNormalization(),
        Dropout(0.5),

        # Capas densas
        Dense(64, activation='relu'),
        Dense(output_length, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Arquitectura v1 (guardada para revertir) ──────────────────────────────────
# def get_model(max_length_frames, output_length):
#     model = Sequential([
#         LSTM(64, return_sequences=True,
#              input_shape=(max_length_frames, LENGTH_KEYPOINTS),
#              kernel_regularizer=l2(0.01)),
#         Dropout(0.5),
#         LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)),
#         Dropout(0.5),
#         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
#         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
#         Dense(output_length, activation='softmax'),
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
