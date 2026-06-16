import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from helpers import mediapipe_detection, there_hand, extract_keypoints, draw_keypoints, get_word_ids, normalize_keypoints, normalize_frame
from constants import MODEL_PATH, WORDS_JSON_PATH, MIN_LENGTH_FRAMES, MODEL_FRAMES

def evaluate_model(src=None, threshold=0.8):
    kp_seq     = []
    sentence   = []
    word_ids   = get_word_ids(WORDS_JSON_PATH)
    model      = load_model(MODEL_PATH)
    count_frame = 0
    fix_frames  = 0
    recording   = False
    MARGIN      = 1
    DELAY       = 3

    with Holistic(model_complexity=0, enable_segmentation=False) as holistic:
        video = cv2.VideoCapture(src or 0)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)

            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > MARGIN:
                    # normalize_frame: el modelo se entrena con datos normalizados
                    kp_seq.append(normalize_frame(extract_keypoints(results)))
            else:
                if count_frame >= MIN_LENGTH_FRAMES + MARGIN:
                    fix_frames += 1
                    if fix_frames < DELAY:
                        recording = True
                        continue
                    seq        = kp_seq[:-(MARGIN + DELAY)]
                    normalized = normalize_keypoints(seq, MODEL_FRAMES)
                    res        = model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
                    if res[np.argmax(res)] > threshold:
                        word = word_ids[int(np.argmax(res))]
                        sentence.insert(0, word)
                recording   = False
                fix_frames  = 0
                count_frame = 0
                kp_seq      = []

            if not src:
                draw_keypoints(frame, results)
                cv2.rectangle(frame, (0,0), (640,35), (245,117,16), -1)
                cv2.putText(frame, ' | '.join(sentence), (5,30),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
                cv2.imshow('SeñAlerta — Reconocimiento', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        video.release()
        cv2.destroyAllWindows()
        return sentence

if __name__ == '__main__':
    evaluate_model()
