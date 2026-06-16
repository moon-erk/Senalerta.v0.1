import os

MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS  = 1662
MODEL_FRAMES      = 15

APPDATA_PATH       = os.path.join(os.environ['LOCALAPPDATA'], 'SenAlerta')
FRAME_ACTIONS_PATH = os.path.join(APPDATA_PATH, 'frame_actions')
DATA_PATH          = os.path.join(APPDATA_PATH, 'data')
KEYPOINTS_PATH     = os.path.join(DATA_PATH, 'keypoints')
MODEL_FOLDER_PATH  = os.path.join(APPDATA_PATH, 'models')
MODEL_PATH         = os.path.join(MODEL_FOLDER_PATH, f'actions_{MODEL_FRAMES}.keras')
WORDS_JSON_PATH    = os.path.join(MODEL_FOLDER_PATH, 'words.json')
SIGNS_METADATA_PATH = os.path.join(MODEL_FOLDER_PATH, 'signs_metadata.json')

# Captura de letras (posturas estáticas): frames por muestra y frames de
# estabilidad antes de empezar a capturar
LETTER_SAMPLE_FRAMES = 7
LETTER_STABLE_FRAMES = 2

EMOTIONS_LIST      = ['feliz', 'triste', 'neutral', 'enojado', 'sorprendido']
EMOTIONS_PATH      = os.path.join(DATA_PATH, 'emotions')
EMOTION_MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, 'emotion_model.keras')
EMOTION_WORDS_PATH = os.path.join(MODEL_FOLDER_PATH, 'emotions.json')

for p in [FRAME_ACTIONS_PATH, KEYPOINTS_PATH, MODEL_FOLDER_PATH, EMOTIONS_PATH]:
    os.makedirs(p, exist_ok=True)
