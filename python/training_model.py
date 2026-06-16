"""
training_model.py — Entrenamiento del modelo LSTM de señas.

Mejoras incluidas en esta versión:
  - Normalización por posición/escala del cuerpo (normalize_frame)
  - Aumento de datos × 5 por muestra real (solo al set de entrenamiento;
    el split train/val se hace antes, sobre muestras reales)
  - Arquitectura mejorada (ver model.py)
  - Matriz de confusión guardada en models/confusion_matrix.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import get_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical

from helpers import get_word_ids, get_sequences_and_labels, normalize_frame, normalize_keypoints
from augmentation import augment_sequence
from constants import MODEL_FRAMES, MODEL_PATH, WORDS_JSON_PATH, MODEL_FOLDER_PATH


def _apply_normalize(sequences):
    """Aplica normalize_frame a cada frame de cada secuencia."""
    normalized = []
    for seq in sequences:
        normalized.append([normalize_frame(frame) for frame in seq])
    return normalized


def _augment_all(sequences, labels, factor=5):
    """
    Genera `factor` variaciones aumentadas por cada muestra real.
    Devuelve las secuencias originales + las aumentadas, con sus etiquetas.
    """
    aug_seqs, aug_labels = [], []
    for seq, label in zip(sequences, labels):
        variations = augment_sequence(seq, factor)
        aug_seqs.extend(variations)
        aug_labels.extend([label] * factor)
    return sequences + aug_seqs, labels + aug_labels


def _save_confusion_matrix(model, X_val, y_val, word_ids, out_path):
    """Genera y guarda la matriz de confusión como PNG."""
    try:
        y_pred = model.predict(X_val, verbose=0)
        y_true_idx = np.argmax(y_val, axis=1)
        y_pred_idx = np.argmax(y_pred, axis=1)

        # labels explícitos: evita fallo cuando el set de validación es muy
        # pequeño y no contiene todas las clases
        cm = confusion_matrix(y_true_idx, y_pred_idx,
                              labels=list(range(len(word_ids))))
        fig, ax = plt.subplots(figsize=(max(6, len(word_ids)), max(5, len(word_ids))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=word_ids)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title('Matriz de confusión — Señas LSC', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Matriz de confusión guardada: {out_path}')
    except Exception as e:
        print(f'No se pudo guardar la matriz de confusión: {e}')


def training_model(model_path=MODEL_PATH, epochs=500, augment_factor=5):
    """
    Entrena el modelo LSTM con aumento de datos y normalización corporal.

    Args:
        model_path:     ruta donde se guarda el modelo .keras
        epochs:         máximo de épocas (EarlyStopping puede detener antes)
        augment_factor: variaciones aumentadas por muestra (0 = sin aumento)
    """
    word_ids = get_word_ids(WORDS_JSON_PATH)
    if not word_ids:
        raise ValueError('No hay palabras. Crea los keypoints primero.')

    print(f'Cargando secuencias para {len(word_ids)} señas...')
    sequences, labels = get_sequences_and_labels(word_ids)
    print(f'  Muestras reales: {len(sequences)}')

    # 1. Normalización corporal (consistente con evaluación en tiempo real)
    print('Aplicando normalización corporal...')
    sequences = _apply_normalize(sequences)

    # 2. Split train/validación SOBRE MUESTRAS REALES, antes del aumento.
    #    Así la validación nunca contiene variaciones de muestras que el
    #    modelo vio en entrenamiento, y su precisión refleja la realidad.
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.1, random_state=42, stratify=labels
    )

    # 3. Aumento de datos SOLO en el set de entrenamiento
    if augment_factor > 0:
        print(f'Aplicando aumento de datos (×{augment_factor}) solo a train...')
        train_seqs, train_labels = _augment_all(train_seqs, train_labels, augment_factor)
        print(f'  Train con aumentados: {len(train_seqs)}')

    # 4. Ajuste temporal a 15 frames con normalize_keypoints (interpolación),
    #    el MISMO método que usa la evaluación en tiempo real — no pad/truncate
    def _to_tensors(seqs, lbls):
        resampled = [normalize_keypoints(list(seq), int(MODEL_FRAMES)) for seq in seqs]
        X = np.array(resampled, dtype=np.float32)
        y = to_categorical(lbls, num_classes=len(word_ids)).astype(int)
        return X, y

    X_train, y_train = _to_tensors(train_seqs, train_labels)
    X_val,   y_val   = _to_tensors(val_seqs, val_labels)
    print(f'Train: {len(X_train)} | Val: {len(X_val)} (solo muestras reales)')

    # 5. Construcción y entrenamiento del modelo
    model = get_model(int(MODEL_FRAMES), len(word_ids))
    model.summary()

    # monitor val_loss (continuo): con sets de validación pequeños la
    # val_accuracy se satura en 1.0 desde la época 1 y restore_best_weights
    # devolvería un modelo sin entrenar. start_from_epoch impide que las
    # primeras épocas (ruido inicial) queden como "mejor" modelo.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        start_from_epoch=15,
        verbose=1
    )

    # Pesos de clase: compensa el desbalance entre señas con muchas y pocas
    # muestras (sin esto, la seña mayoritaria domina el aprendizaje)
    classes = np.unique(train_labels)
    cw = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )

    # 6. Reporte de precisión
    train_acc = max(history.history.get('accuracy', [0]))
    val_acc   = max(history.history.get('val_accuracy', [0]))
    print(f'\n=== Resultados finales ===')
    print(f'  Accuracy entrenamiento : {train_acc:.4f}  ({train_acc*100:.1f}%)')
    print(f'  Accuracy validación    : {val_acc:.4f}  ({val_acc*100:.1f}%)')

    # 7. Matriz de confusión
    cm_path = os.path.join(MODEL_FOLDER_PATH, 'confusion_matrix.png')
    _save_confusion_matrix(model, X_val, y_val, word_ids, cm_path)

    # 8. Guardar modelo
    model.save(model_path)
    print(f'Modelo guardado: {model_path}')

    return train_acc, val_acc


if __name__ == '__main__':
    train_acc, val_acc = training_model()
    print(f'\nAccuracy train={train_acc:.4f}  val={val_acc:.4f}')
