# CLAUDE.md — SeñAlerta

Este archivo lo lees al inicio de cada sesión. Contiene todo lo que necesitas saber sobre el proyecto. Léelo completo antes de trabajar.

\---

## QUÉ ES SEÑALERTA

Aplicación de escritorio y móvil que traduce Lengua de Señas Colombiana (LSC) a voz, y voz a texto, para permitir una conversación completa entre una persona sorda y una persona oyente. El usuario principal es una persona sorda comunicándose cara a cara con un oyente.

Eslogan: "Conectando voces, rompiendo barreras."
Creado en la Institución Educativa Siglo XXI de Tauramena, Casanare.
Colores de marca: Turquesa #18B7B0, Naranja Coral #F26B4A, Blanco #FFFFFF.

\---

## EL CASO DE USO CENTRAL — SESIONES POR QR

La persona sorda tiene la app instalada con todas las funciones. Para conversar con un oyente:

1. La persona sorda inicia una sesión → se genera un código QR
2. El oyente escanea el QR con su teléfono → se abre una página web ligera (sin instalar nada)
3. La persona sorda hace señas → el oyente recibe el texto Y lo escucha en voz (TTS)
4. El oyente responde escribiendo o hablando → le llega como texto a la persona sorda

Es un chat en vivo entre dos dispositivos unidos por QR. Los mensajes viajan por Supabase Realtime.

\---

## ENTORNO DE PYTHON — CRÍTICO

* Versión correcta: Python 3.10.11 (NO 3.11, NO 3.9)
* El entorno virtual está en `C:\\\\v310\\\\`
* SIEMPRE usa `C:\\\\v310\\\\Scripts\\\\python.exe` para ejecutar Python
* SIEMPRE usa `C:\\\\v310\\\\Scripts\\\\pip.exe` para instalar paquetes
* Nunca uses el Python del sistema ni otro entorno virtual

Razón: es la única versión donde mediapipe 0.10.11, tensorflow-cpu 2.15.1 y tables 3.9.2 instalan sin conflicto en Windows.

\---

## CÓMO ARRANCAR EL BACKEND

```powershell
C:\\\\v310\\\\Scripts\\\\python.exe python/server.py
```

Cuando imprime "Backend listo en http://localhost:8765" está listo. Corre con Waitress en el puerto 8765.

Antes de arrancar, si el puerto 8765 está ocupado:

```powershell
netstat -ano | findstr 8765
taskkill /PID \\\[numero] /F
```

\---

## EL BACKEND YA FUNCIONA — NO LO ROMPAS

El sistema de reconocimiento está construido y probado end-to-end. Hace:

* Cámara con stream MJPEG (`/camera/stream`)
* Detección de landmarks con MediaPipe Holistic (manos, cara, cuerpo)
* Captura automática de muestras (detecta cuándo entran las manos)
* Creación de keypoints en HDF5
* Entrenamiento de red LSTM con TensorFlow
* Reconocimiento en tiempo real con barra de confianza

NO modifiques el backend a menos que se pida explícitamente. Si necesitas cambiarlo, primero explica qué vas a tocar y por qué.

\---

## ESTRUCTURA DEL PROYECTO

```
señalerta/
├── python/
│   ├── constants.py       # rutas y constantes (MODEL\\\_FRAMES=15, LENGTH\\\_KEYPOINTS=1662)
│   ├── helpers.py         # extract\\\_keypoints, normalize\\\_keypoints, draw\\\_keypoints
│   ├── model.py           # arquitectura LSTM
│   ├── training\\\_model.py  # entrenamiento
│   ├── capture\\\_samples.py # captura
│   ├── evaluate\\\_model.py  # evaluación
│   ├── text\\\_to\\\_speech.py  # gTTS + pygame
│   └── server.py          # servidor Flask, único punto de entrada
├── test\\\_popup.html        # popup de prueba standalone (se abre en Chrome)
├── src/                   # frontend React (Tauri) — diseño visual
└── src-tauri/             # configuración Tauri
```

Los datos del usuario se guardan en `C:\\\\Users\\\\\\\[usuario]\\\\AppData\\\\Local\\\\SeñAlerta\\\\`:

* `frame\\\_actions/` — imágenes JPG de las muestras
* `data/keypoints/` — archivos HDF5 por palabra
* `models/` — modelo entrenado (actions\_15.keras) y words.json

\---

## ARQUITECTURA TÉCNICA — CUATRO PIEZAS

1. App de la persona sorda (frontend React, adaptable móvil/escritorio)
2. Backend local de reconocimiento (Python + Flask, localhost:8765) — YA HECHO
3. Página web del oyente (web ligera, se abre por QR)
4. Supabase (nube) — cuentas opcionales, estadísticas, canal de tiempo real para el QR

\---

## CONSTANTES TÉCNICAS DEL MODELO

* MODEL\_FRAMES = 15 (frames por seña)
* LENGTH\_KEYPOINTS = 1662 (33×4 pose + 468×3 cara + 21×3 mano izq + 21×3 mano der)
* MIN\_LENGTH\_FRAMES = 5
* THRESHOLD = 0.8 (umbral de confianza para confirmar una seña)
* Arquitectura: LSTM(64) → Dropout(0.5) → LSTM(128) → Dropout(0.5) → Dense(64) → Dense(64) → Softmax

\---

## REGLAS DE TRABAJO

1. No toques el diseño visual del frontend a menos que se pida explícitamente.
2. No modifiques el backend de reconocimiento sin avisar primero qué y por qué.
3. Cuando algo falle, copia el error COMPLETO de la terminal, sin resumir ni parafrasear.
4. Antes de dar por terminado algo, verifícalo de verdad (ejecuta, prueba, confirma).
5. El test\_popup.html debe seguir siendo HTML puro standalone, sin dependencias de build.
6. Usa siempre las rutas de AppData para datos de usuario, nunca la carpeta del proyecto.
7. La interfaz debe ser extremadamente adaptable: nada de tamaños fijos que rompan el layout. Todo se adapta de móvil a escritorio.

\---

## ESTADO ACTUAL DEL PROYECTO

* Backend de reconocimiento: terminado y probado
* test\_popup.html: funcionando, con diseño inspirado en "Echo's Glasses"
* Documento de arquitectura funcional: definido (ver ARQUITECTURA\_FUNCIONAL.md)
* Diagramas de flujo: hechos
* Próximo paso: diseño visual de las pantallas finales, luego integración en Tauri, luego sistema de cuentas con Supabase

\---

## DECISIONES PENDIENTES

* Plataforma móvil: ¿app nativa de tienda o web app que se comporta como app?
* Qué pantalla ve primero el usuario que ya tiene cuenta y pasó la bienvenida
* Si las conversaciones se guardan o se borran al cerrar la sesión

## DOCUMENTACIÓN DEL PROYECTO
Toda la documentación, decisiones y progreso del proyecto están en la carpeta /docs.
Antes de trabajar, consulta /docs/06-progreso.md para saber el estado actual.
Para contexto de arquitectura, consulta /docs/DOCUMENTO_MAESTRO.md.
Cuando termines algo importante, actualiza /docs/06-progreso.md.
