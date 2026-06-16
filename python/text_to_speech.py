import os
import tempfile
import threading
from time import sleep

from gtts import gTTS
import pygame

# pygame no es seguro entre hilos: init/quit concurrentes desde varios hilos
# (una seña reconocida tras otra) tumban el proceso en código nativo, y el
# archivo compartido speech.mp3 se pisaba entre reproducciones.
# Solución: un lock global serializa el TTS, el mixer se inicializa UNA vez,
# y cada reproducción usa un archivo temporal propio.
_tts_lock    = threading.Lock()
_mixer_ready = False


def text_to_speech(text):
    global _mixer_ready
    with _tts_lock:
        tts = gTTS(text=text, lang='es')
        fd, filename = tempfile.mkstemp(suffix='.mp3', prefix='senalerta_tts_')
        os.close(fd)
        try:
            tts.save(filename)
            if not _mixer_ready:
                pygame.mixer.init()
                _mixer_ready = True
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                sleep(0.1)
            pygame.mixer.music.unload()
        finally:
            try:
                os.remove(filename)
            except OSError:
                pass
