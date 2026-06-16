import os
import cv2
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from helpers import mediapipe_detection, there_hand, draw_keypoints, save_frames
from constants import FRAME_ACTIONS_PATH, MIN_LENGTH_FRAMES

def capture_samples(word_id, n_samples=10):
    word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
    os.makedirs(word_path, exist_ok=True)

    count      = 0
    frames     = []
    recording  = False
    fix_frames = 0
    MARGIN     = 1
    DELAY      = 3

    with Holistic(model_complexity=0, enable_segmentation=False) as holistic:
        cap = cv2.VideoCapture(0)
        print(f'Capturando muestras para "{word_id}". Muestra 1/{n_samples}. Presiona Q para salir.')

        while cap.isOpened() and count < n_samples:
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)
            draw_keypoints(frame, results)

            if there_hand(results) or recording:
                recording = False
                frames.append(frame.copy())
            else:
                if len(frames) >= MIN_LENGTH_FRAMES + MARGIN:
                    fix_frames += 1
                    if fix_frames < DELAY:
                        recording = True
                    else:
                        today       = datetime.now().strftime('%y%m%d%H%M%S%f')
                        sample_path = os.path.join(word_path, f'sample_{today}')
                        os.makedirs(sample_path, exist_ok=True)
                        save_frames(frames[:-(MARGIN + DELAY)], sample_path)
                        count += 1
                        print(f'Muestra {count}/{n_samples} guardada')
                        frames     = []
                        fix_frames = 0
                        recording  = False

            cv2.putText(frame, f'{word_id}: {count}/{n_samples}', (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
            cv2.imshow('Captura de muestras', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    print(f'Captura terminada: {count} muestras guardadas en {word_path}')

if __name__ == '__main__':
    import sys
    word = sys.argv[1] if len(sys.argv) > 1 else input('Nombre de la seña: ').strip()
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    capture_samples(word, n)
