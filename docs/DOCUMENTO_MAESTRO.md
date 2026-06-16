# SEÑALERTA — DOCUMENTO MAESTRO DE DESARROLLO
## Todas las decisiones de backend, frontend y UI/UX

Este documento consolida todas las decisiones tomadas para desarrollar SeñAlerta por completo. Es la fuente de verdad para los prompts que se le pasan a Claude Code.

---

## PARTE A — BACKEND

### Alcance de esta versión
- Backend Python local para escritorio (el que ya funciona, se mantiene y mejora)
- Preparación de la versión móvil con TensorFlow Lite (modelo nuevo, más adelante)
- Integración de Supabase: cuentas, estadísticas, canal de tiempo real para QR
- Solo reconocimiento de señas por ahora (la emoción ya está planeada aparte)

### Arquitectura de datos y privacidad (CRÍTICO)
Esta es la decisión más importante del proyecto:

- **Reconocimiento de señas (local):** usa el vector completo de 1662 valores, incluyendo los 468 puntos faciales, para máxima precisión. Todo el reconocimiento ocurre en el dispositivo.
- **Donación de señas para mejorar el modelo común:** cuando el usuario acepta contribuir, se sube SOLO el esqueleto de cuerpo y manos (sin los puntos faciales). Esto es anónimo y de bajo riesgo legal porque el esqueleto de una seña no identifica a la persona. Requiere consentimiento explícito.
- **Modelo de emoción facial:** usa los keypoints faciales detallados, pero NUNCA salen del dispositivo o cuenta del usuario. Es estrictamente local.

Justificación legal: bajo la Ley 1581 de Colombia, los datos biométricos (como la geometría facial) son sensibles. El esqueleto de cuerpo/manos no identifica a una persona, por eso se puede compartir anónimamente; la cara sí, por eso se queda local.

### Modelo base y personalización
- La app trae de fábrica un modelo base con: saludos básicos + abecedario completo. Igual para todos al instalar.
- El usuario puede editar las señas base (regrabar para mejorar) y añadir señas nuevas.
- Al editar o añadir, se reentrena localmente una copia personalizada (base + señas del usuario).
- El modelo base original se mantiene; cada usuario tiene su versión personalizada en su dispositivo.
- Opcionalmente, el usuario puede donar sus señas (solo esqueleto, anónimo, con consentimiento) para mejorar el modelo común en futuras versiones.

### Sesiones QR
- El backend manda a Supabase el texto reconocido + la frase armada.
- Nota de arquitectura a resolver con Claude Code: evaluar si es el backend Python o el frontend quien habla con Supabase (normalmente el frontend es mejor por estar en pantalla y manejar la sesión visual).

### Texto a voz (TTS)
- En la app de la persona sorda: TTS local (gTTS, como ahora).
- En la página del oyente: voz del navegador (Web Speech API).

### Arranque del backend
- Por ahora: manual (el usuario lo arranca).
- Después: sidecar de Tauri (arranca solo al abrir la app).

### Versión móvil (preparación)
- Se entrenará un modelo nuevo pensado para móvil en formato TensorFlow Lite.
- Correrá dentro del teléfono con MediaPipe Tasks + TF Lite, sin internet.
- Requiere capturar/preparar datos propios para ese modelo.

---

## PARTE B — FRONTEND

### Tecnología
- Empezar el frontend de cero, más ordenado.
- Proyecto totalmente nuevo, migrando lo que sirve del actual.
- React + Tauri para escritorio.
- React Native (o equivalente) para móvil, más adelante.
- Navegación con React Router entre las 7 pantallas.

### Base visual
- Mezclar: la estructura y disposición de Echo's Glasses (paneles, terminal log, HUDs, widget de predicción, panel lateral) con los colores del manual de marca de SeñAlerta (turquesa #18B7B0, naranja coral #F26B4A, blanco).

### Modo de color
- Solo modo claro por ahora (oscuro después).

### Las 7 pantallas
1. **Bienvenida** — primera vez. Qué hace la app, permisos de cámara/micrófono.
2. **Cuenta** — opcional, con "continuar sin cuenta". Conecta a Supabase.
3. **Inicio / Resumen** — lo que ve el usuario recurrente. Accesos rápidos (iniciar conversación, aprender, biblioteca) + resumen de uso si tiene cuenta.
4. **Conversar** — la principal. Cámara, reconocimiento, barra de confianza, historial de conversación, QR para que el oyente se una.
5. **Aprender** — capturar señas nuevas, editar las base, crear keypoints, entrenar.
6. **Biblioteca** — lista de señas conocidas, con sus muestras, añadir/eliminar.
7. **Ajustes** — cámara, voz TTS, umbral de confianza, cuenta, donación de señas (con consentimiento).

---

## PARTE C — UI / UX

### Usuario principal
Persona sorda comunicándose con oyentes. Esto guía todas las decisiones.

### Prioridad visual
- Equilibrio: interfaz limpia pero con información de apoyo visible (no minimalista extremo, no recargada).

### Botones y acciones
- Iconos grandes + texto siempre visible. Nunca depender solo de iconos.

### Accesibilidad (fuerte, es prioridad)
- Contraste alto en todos los textos y controles.
- Texto grande y escalable.
- Foco visible claro (para navegación por teclado).
- Todo lo importante comunicado visualmente, nunca solo por sonido.

### Animaciones
- Sutiles, que guíen al usuario (transiciones suaves entre estados y pantallas).
- No exageradas ni distractoras.

### Retroalimentación
- Visual fuerte: como la app es para personas sordas, todo evento (seña reconocida, error, conexión del oyente, etc.) debe avisarse claramente de forma visual — cambios de color, animaciones, indicadores grandes. Nunca depender de sonido para avisar algo importante.

---

## PARTE D — ORDEN DE CONSTRUCCIÓN SUGERIDO

### Etapa 1 — Frontend nuevo y ordenado
1. Crear proyecto nuevo React + Tauri + React Router
2. Definir el sistema de diseño: colores de marca, tipografía, componentes base (botones, paneles, inputs) con accesibilidad fuerte
3. Construir las 7 pantallas con navegación, adaptables a cualquier tamaño
4. Migrar la lógica de conexión al backend que ya funciona

### Etapa 2 — Conectar el reconocimiento
5. Conectar la pantalla Conversar al backend (stream, landmarks, confianza, frase)
6. Conectar Aprender y Biblioteca al backend (captura, keypoints, entrenamiento)

### Etapa 3 — Modelo base y personalización
7. Crear el modelo base de fábrica (saludos + abecedario)
8. Sistema de edición de señas base y añadido de señas nuevas
9. Reentrenamiento local de la copia personalizada

### Etapa 4 — Supabase
10. Cuentas opcionales (registro/login)
11. Sesiones QR en tiempo real (app ↔ página del oyente)
12. Página web del oyente (ligera, por QR)
13. Estadísticas de uso
14. Donación de señas (solo esqueleto, con consentimiento) — puede quedar para después

### Etapa 5 — Empaquetado y móvil
15. Sidecar de Tauri (backend arranca solo)
16. Instalador de escritorio
17. Modelo TF Lite y app móvil nativa

---

## NOTAS IMPORTANTES PARA TODO EL DESARROLLO

- El reconocimiento de señas actual funciona y está probado. No romperlo.
- Entorno Python: C:\v310\Scripts\python.exe (Python 3.10.11).
- Backend en localhost:8765.
- La cuenta de Supabase la crea el equipo (gratis); las claves se pegan en la config.
- Yo (Claude en chat) no puedo crear cuentas reales ni manejar credenciales; eso lo hace el equipo. Claude Code escribe el código; el equipo configura los servicios externos.
- La donación de datos biométricos requiere consentimiento explícito e informado, especial cuidado con menores. Solo esqueleto anónimo, nunca cara.
