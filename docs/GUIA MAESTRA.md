

## La lista de pasos en orden, con los prompts listos para copiar y pegar

No tienes que pensar en todo a la vez. Solo sigue esta lista en orden. Cada paso tiene su prompt listo para pegarle a Claude Code. Haces un paso, verificas que funciona, y pasas al siguiente. Nada más.

---

## CÓMO USAR ESTA GUÍA

1. Abre Claude Code en la carpeta del proyecto
2. Copia el prompt del paso en el que vas
3. Pégalo y deja que trabaje
4. Verifica lo que te diga la sección "CÓMO VERIFICAR" del paso
5. Anota en Obsidian (docs/06-progreso.md) que terminaste ese paso
6. Sigue con el siguiente

Si algo falla en cualquier paso: copia el error completo y pégamelo a mí (Claude en el chat) para ayudarte a resolverlo antes de continuar.

---

# FASE A — ORGANIZAR LA CASA (1 sesión)

## Paso A1 — Verificar que la memoria está conectada

**Prompt para Claude Code:**

```
Lee el CLAUDE.md y la carpeta /docs completa. Dime en 5 líneas: qué es este proyecto, en qué estado está, y cuál es el siguiente paso según la documentación. No hagas nada más.
```

**Cómo verificar:** Claude Code debe responder describiendo SeñAlerta correctamente (app de señas, QR, backend funcionando). Si no encuentra /docs o el CLAUDE.md, la memoria no está bien conectada — avísame.

---

## Paso A2 — Documentar la red neuronal en Obsidian

Esto crea las notas que explican TU red neuronal, para que entiendas y puedas presentar lo que hace.

**Prompt para Claude Code:**

```
Lee python/model.py, python/training_model.py y python/helpers.py. Luego crea en /docs una carpeta llamada "red-neuronal" con estas notas en markdown, explicadas en lenguaje sencillo para un estudiante:

1. que-es-el-lstm.md — qué es una red LSTM y por qué este proyecto la usa para señas (las señas son secuencias de movimiento en el tiempo)
2. flujo-de-datos.md — el camino completo: cámara → MediaPipe → 1662 keypoints por frame → secuencia de 15 frames → LSTM → predicción con softmax
3. arquitectura-del-modelo.md — capa por capa qué hace el modelo actual (LSTM 64, Dropout, LSTM 128, Dense...) con explicación simple de cada concepto (qué es dropout, qué es softmax, qué es una época)
4. bitacora-experimentos.md — una plantilla de tabla para anotar cada entrenamiento: fecha, número de señas, muestras por seña, precisión obtenida, cambios hechos

Usa español sencillo, sin asumir conocimientos de machine learning.
```

**Cómo verificar:** Abre Obsidian y confirma que la carpeta "red-neuronal" existe con las 4 notas. Léelas — deberías poder entenderlas.

---

# FASE B — POTENCIAR LA RED NEURONAL (2-3 sesiones)

## Paso B1 — Aumento de datos

**Prompt para Claude Code:**

```
Lee CLAUDE.md y /docs/PLAN_IA_PROPIA.md. Implementa SOLO la parte de aumento de datos:

Crea python/augmentation.py con una función que tome una secuencia de keypoints (15 frames × 1662 valores) y genere N variaciones aplicando: rotación leve (-10 a +10 grados en XY), desplazamiento (±0.05), escalado (0.9-1.1), ruido gaussiano (desviación 0.01), y espejo horizontal (invertir x e intercambiar manos izquierda/derecha).

Luego modifica training_model.py para aplicar el aumento multiplicando cada muestra por 5 antes de entrenar.

No toques nada más. Cuando termines, reentrena con los datos actuales y dime la precisión antes y después del aumento.
```

**Cómo verificar:** Te debe dar dos números de precisión. Si la precisión con aumento es igual o mejor, perfecto. Anota los números en docs/red-neuronal/bitacora-experimentos.md.

---

## Paso B2 — Mejor normalización

**Prompt para Claude Code:**

```
En python/helpers.py añade una función normalize_frame que normalice cada frame de 1662 valores relativo al cuerpo: el punto medio entre los hombros (pose landmarks 11 y 12) como origen, y la distancia entre hombros como escala. Aplícala de forma consistente en el entrenamiento Y en la evaluación en tiempo real (server.py). Esto hace que las señas se reconozcan igual sin importar la distancia o posición frente a la cámara.

Reentrena y dime la precisión. Compárala con la del paso anterior.
```

**Cómo verificar:** El backend sigue reconociendo señas (pruébalo en el popup) y te da la nueva precisión. Anótala en la bitácora.

---

## Paso B3 — Matriz de confusión

**Prompt para Claude Code:**

```
Modifica el entrenamiento para que al terminar genere una matriz de confusión (qué señas se confunden entre sí) y la guarde como imagen PNG en AppData/Local/SeñAlerta/models/confusion_matrix.png usando matplotlib. Entrena y muéstrame la ruta de la imagen generada.
```

**Cómo verificar:** Abre la imagen PNG. Verás una cuadrícula donde la diagonal debe ser la más marcada (señas bien reconocidas). Si una seña se confunde mucho con otra, necesita más muestras — anótalo.

---

# FASE C — EL MODELO BASE DE FÁBRICA (2-4 sesiones + tu tiempo grabando)

## Paso C1 — Capturar las señas base

Esto lo haces TÚ frente a la cámara, con el popup. Las señas base que la app traerá de fábrica:

- Saludos: hola, gracias, por favor, sí, no, buenos días, adiós, cómo estás
- Abecedario: A, B, C... (las 27 letras)

Para cada una: 15-20 muestras mínimo. Es trabajo repetitivo pero es EL corazón del proyecto. Puedes hacerlo en varias sesiones (un día los saludos, otro día letras A-H, etc.).

**Cómo verificar:** En el popup, la lista de muestras muestra cada seña con 15+ muestras.

---

## Paso C2 — Entrenar el modelo base

**Prompt para Claude Code:**

```
Ya capturé todas las señas base (saludos + abecedario). Ejecuta el flujo completo: crear keypoints de todas las señas, entrenar el modelo con el aumento de datos y la normalización ya implementados, generar la matriz de confusión, y dime la precisión final y qué señas se confunden entre sí según la matriz.
```

**Cómo verificar:** Precisión por encima del 85% es buena para empezar. Si alguna seña se confunde mucho, grábale más muestras y reentrena.

---

# FASE D — EL FRONTEND NUEVO (4-6 sesiones)

## Paso D1 — Crear el proyecto nuevo

**Prompt para Claude Code:**

```
Lee /docs/DOCUMENTO_MAESTRO.md secciones B y C. Crea un proyecto frontend NUEVO y ordenado en una carpeta /app dentro del proyecto: React + Vite + React Router + Tauri. Configura el sistema de diseño base: modo claro, colores de marca (turquesa #18B7B0 primario, coral #F26B4A acento, blanco fondo), tipografía legible y grande, componentes base accesibles (Button con icono+texto siempre, Card, Input) con contraste alto y foco visible. Estructura de carpetas ordenada: /app/src/screens (las 7 pantallas vacías por ahora), /app/src/components, /app/src/lib (conexión al backend). Solo el esqueleto y el sistema de diseño — las pantallas se construyen en los siguientes pasos. Verifica que arranca con npm run dev sin errores.
```

**Cómo verificar:** `npm run dev` abre la app vacía con la navegación entre 7 pantallas funcionando.

---

## Paso D2 — Pantalla Conversar (la principal)

**Prompt para Claude Code:**

```
Construye la pantalla Conversar en /app: vista de cámara grande (stream MJPEG de localhost:8765/camera/stream), barra de confianza con la palabra actual, historial de conversación, botón grande "Iniciar conversación" que por ahora muestra un QR de ejemplo (la sesión real con Supabase viene después). Estructura tipo Echo's Glasses pero en modo claro con los colores de marca. Retroalimentación visual fuerte: cuando se reconoce una seña, la palabra aparece con una animación clara. Todo adaptable: en pantalla ancha cámara + panel lateral, en estrecha apilado. Conéctala al backend real y verifica que reconoce señas.
```

**Cómo verificar:** Abres la app, enciendes cámara, haces una seña conocida y aparece reconocida con su animación.

---

## Paso D3 — Pantallas Aprender y Biblioteca

**Prompt para Claude Code:**

```
Construye las pantallas Aprender y Biblioteca en /app conectadas al backend:

Aprender: flujo guiado para añadir una seña nueva — escribir nombre, capturar muestras (con contador en vivo), botón para crear keypoints y entrenar con barra de progreso. Usa los endpoints existentes (/capture/start, /capture/stop, /capture/samples, /keypoints/create, /train, /train/status).

Biblioteca: lista de todas las señas del modelo con sus muestras, opción de añadir más muestras a una seña existente y de eliminar señas (/capture/delete). Distinguir visualmente las señas base de fábrica de las añadidas por el usuario.

Mismo sistema de diseño, iconos+texto, accesible, adaptable.
```

**Cómo verificar:** Puedes añadir una seña nueva desde la app, entrenarla, y verla en la Biblioteca.

---

## Paso D4 — Pantallas restantes (Bienvenida, Cuenta, Inicio, Ajustes)

**Prompt para Claude Code:**

```
Construye las 4 pantallas restantes en /app:

Bienvenida: solo primera vez, explica la app, pide permisos de cámara/micrófono.
Cuenta: formulario de registro/login con botón claro "Continuar sin cuenta" — por ahora sin conectar a Supabase, solo la pantalla lista con los puntos de conexión marcados con TODO.
Inicio/Resumen: accesos rápidos grandes (Iniciar conversación, Aprender, Biblioteca) — es lo primero que ve el usuario recurrente.
Ajustes: selector de cámara, voz TTS, umbral de confianza, sección de cuenta.

El flujo de navegación: primera vez → Bienvenida → Cuenta → Inicio. Usuario recurrente → directo a Inicio.
```

**Cómo verificar:** Navegas todo el flujo completo de pantallas sin errores.

---

# FASE E — SUPABASE Y LAS SESIONES QR (3-4 sesiones)

## Paso E1 — Crear la cuenta de Supabase (LO HACES TÚ)

Esto no lo puede hacer Claude Code ni yo. Tú:

1. Ve a https://supabase.com y crea una cuenta gratis con tu correo
2. Crea un proyecto nuevo llamado "senalerta"
3. En el panel del proyecto, ve a Settings → API
4. Copia dos cosas: la "Project URL" y la "anon public key"
5. Guárdalas en un archivo de notas — las necesitas para el siguiente paso

---

## Paso E2 — Conectar cuentas y sesiones QR

**Prompt para Claude Code (reemplaza URL_AQUI y KEY_AQUI con tus claves):**

```
Integra Supabase en /app. URL del proyecto: URL_AQUI — Anon key: KEY_AQUI (ponlas en un archivo de configuración .env, no en el código).

1. Conecta la pantalla Cuenta al auth de Supabase (registro/login con correo, y continuar sin cuenta sigue funcionando).
2. Implementa las sesiones QR: al presionar "Iniciar conversación" se crea un canal de Supabase Realtime con un ID único, y el QR codifica la URL de la página del oyente con ese ID.
3. Crea la página del oyente como proyecto web separado en /oyente: página ligera que se conecta al canal del ID en la URL, muestra los mensajes de la persona sorda en texto grande y los lee en voz con la Web Speech API del navegador, y permite responder escribiendo o hablando (reconocimiento de voz del navegador). Los mensajes del oyente llegan al historial de la app.
4. La app envía al canal cada frase armada que se reconoce.
```

**Cómo verificar:** Abres la app, inicias conversación, escaneas el QR con tu celular, se abre la página del oyente, haces una seña y aparece en el celular con voz. Respondes desde el celular y aparece en la app. ESA es la magia completa del proyecto funcionando.

---

# FASE F — EMPAQUETAR (2-3 sesiones)

## Paso F1 — Sidecar y instalador

**Prompt para Claude Code:**

```
Empaqueta el backend Python con PyInstaller usando C:\v310\Scripts\pyinstaller.exe, configúralo como sidecar de Tauri para que arranque automáticamente al abrir la app, y genera el instalador de Windows con npm run tauri:build. Verifica que la app instalada funciona completa sin que el usuario abra ninguna terminal.
```

**Cómo verificar:** Instalas el .exe en tu PC, abres SeñAlerta desde el menú de inicio, y todo funciona sin terminal.

---

# RESUMEN DEL ORDEN

- FASE A: organizar memoria y documentar la red neuronal (1 sesión)
- FASE B: potenciar la red (aumento de datos, normalización, matriz) (2-3 sesiones)
- FASE C: grabar y entrenar el modelo base de fábrica (saludos + abecedario) (2-4 sesiones + tu grabación)
- FASE D: frontend nuevo con las 7 pantallas (4-6 sesiones)
- FASE E: Supabase, cuentas, QR, página del oyente (3-4 sesiones)
- FASE F: empaquetar instalador (2-3 sesiones)

No pienses en todo. Solo en el paso que sigue. Cada paso terminado se anota en docs/06-progreso.md, y si algo falla, me traes el error a mí.