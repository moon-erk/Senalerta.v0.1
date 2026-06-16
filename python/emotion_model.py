"""
emotion_model.py -- Arquitectura MLP para clasificacion de emociones faciales.

Input:  468 face landmarks x 3 (x,y,z) = 1404 valores
Output: N emociones (softmax)

Arquitectura:
  Dense(128, relu) -> BatchNorm -> Dropout(0.4)
  -> Dense(64, relu) -> Dropout(0.4)
  -> Dense(N, softmax)
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

LENGTH_FACE_KEYPOINTS = 468 * 3  # 1404


def get_emotion_model(n_emotions: int):
    """
    Construye el modelo MLP para clasificacion de emociones.

    Args:
        n_emotions: numero de clases de emocion a reconocer

    Returns:
        Modelo Keras compilado.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(LENGTH_FACE_KEYPOINTS,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(n_emotions, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
