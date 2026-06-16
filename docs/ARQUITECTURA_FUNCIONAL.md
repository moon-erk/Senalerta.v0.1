# SEÑALERTA — ARQUITECTURA FUNCIONAL COMPLETA
## Documento base antes del diseño

Este documento define cómo funciona toda la aplicación. El diseño visual se construye después, sobre esta base. Aquí no hay decisiones de colores ni de cómo se ve nada — solo qué hace la app, qué pantallas tiene, y cómo se conecta todo.

---

## 0. DIAGRAMAS DE FLUJO

Este documento viene acompañado de dos diagramas de flujo (archivos SVG):

- `diagrama_flujo_general.svg` — el recorrido de la app: apertura, bienvenida, cuenta, pantallas principales, y cómo se conectan las cuatro piezas técnicas (app, backend local, página del oyente, Supabase).
- `diagrama_flujo_conversacion.svg` — el corazón funcional: cómo la persona sorda inicia una sesión, el oyente se une por QR, y fluye la conversación bidireccional en vivo.

---

## 1. QUÉ ES SEÑALERTA

Aplicación de escritorio y móvil que traduce Lengua de Señas Colombiana (LSC) a voz, y voz a texto, para permitir una conversación completa entre una persona sorda y una persona oyente. El usuario principal es una persona sorda comunicándose cara a cara con un oyente.

Eslogan de marca: "Conectando voces, rompiendo barreras."

---

## 2. EL CASO DE USO CENTRAL — SESIONES POR QR

El modelo de comunicación es asimétrico y conectado por código QR. Esta es la idea central de la app:

La PERSONA SORDA tiene la app instalada (móvil o escritorio). Ahí están todas las funcionalidades.

Para conversar con un OYENTE, la persona sorda inicia una sesión que genera un código QR. El oyente escanea ese QR con la cámara de su teléfono y se le abre una PÁGINA WEB ligera (sin instalar nada), conectada en vivo a la sesión de la persona sorda.

La conversación fluye en tiempo real entre los dos dispositivos:

- La persona sorda hace señas → la app las reconoce, arma la frase → el oyente recibe en su página web el TEXTO de la frase Y la escucha en VOZ (Text-to-Speech reproducido en su propia página).
- El oyente responde en su página web, donde puede ESCRIBIR texto O HABLAR (su voz se convierte a texto en su propia página) → ese texto le llega a la app de la persona sorda y lo lee en pantalla.

Es como un chat en vivo entre dos dispositivos: uno con la app completa, el otro con una página web ligera, unidos por el QR. El oyente nunca instala nada.

Esto significa que la pantalla de conversación de la app debe ser grande y clara, con el QR fácil de mostrar al inicio de cada sesión.

---

## 3. CÓMO FUNCIONA EL RECONOCIMIENTO DE SEÑAS

Esto ya está construido y probado en el backend Python local (localhost:8765):

- La cámara captura video y MediaPipe Holistic detecta los landmarks de manos, cara y cuerpo
- Una red neuronal LSTM entrenada reconoce qué seña se está haciendo
- El sistema acumula las palabras reconocidas (no las dice una por una)
- Cuando el usuario termina (gesto de palmas abiertas o botón), las palabras acumuladas se arman en una frase con sentido
- La frase final se dice en voz alta con Text-to-Speech

Ejemplo: la persona hace las señas "tecnología, usar, mejorar, vida, no, guerra" → la app las acumula → al terminar arma "Usemos la tecnología para mejorar la vida, no la guerra" → lo dice en voz alta.

---

## 4. EL SISTEMA DE SESIONES POR QR — CÓMO SE CONECTAN LOS DOS DISPOSITIVOS

Esta es la pieza nueva más importante. Permite que dos dispositivos distintos se comuniquen en vivo.

### Cómo arranca una sesión
1. La persona sorda, en su app, presiona "Iniciar conversación"
2. La app crea una sesión en el servidor de tiempo real (Supabase Realtime) y obtiene un código único de sesión
3. La app muestra un código QR que contiene la dirección web de esa sesión
4. El oyente escanea el QR con la cámara de su teléfono
5. Se abre la página web del oyente, ya conectada a esa sesión específica
6. Ambos dispositivos ahora están en el mismo "canal" en tiempo real

### Qué pasa durante la conversación
- Persona sorda hace señas → app reconoce y arma frase → envía el texto al canal → la página del oyente recibe el texto, lo muestra grande Y lo reproduce en voz con Text-to-Speech del navegador
- Oyente escribe o habla en su página → si habla, su voz se convierte a texto en su propio navegador → envía el texto al canal → la app de la persona sorda recibe el texto y lo muestra grande en pantalla

### La página web del oyente
Es una web aparte, muy ligera, alojada en internet. No tiene cámara ni reconocimiento de señas. Solo tiene: un área grande donde aparece lo que dice la persona sorda (texto + voz automática), un campo para escribir, y un botón de micrófono para hablar (que usa el reconocimiento de voz del propio navegador). Funciona en cualquier teléfono sin instalar nada.

### La tecnología detrás
El QR apunta a una dirección en internet, no a localhost. Los mensajes en vivo entre los dos dispositivos viajan por Supabase Realtime — el mismo Supabase que usamos para las cuentas, así que no añadimos otra tecnología. El servidor de tiempo real solo pasa mensajes de texto entre los dos; no guarda el contenido de la conversación.

---

## 5. LAS PANTALLAS DE LA APP

### Pantalla 0 — Bienvenida (solo la primera vez)
Explica brevemente qué hace la app, da la bienvenida, pide permiso de cámara y micrófono. Solo aparece la primera vez que se abre la app en el dispositivo.

### Pantalla 1 — Cuenta (opcional, después de bienvenida)
Aparece después de la bienvenida en un dispositivo nuevo. Ofrece crear cuenta o iniciar sesión con correo y contraseña, PERO tiene un botón claro de "Continuar sin cuenta" o "Más tarde". La cuenta es opcional — la app funciona completa sin ella. Si el usuario crea cuenta, sus estadísticas de uso se guardan asociadas a esa cuenta. Si no, la app funciona igual pero sin guardar nada en la nube.

La cuenta se conecta a Supabase (servicio externo de autenticación). El registro guarda: correo, tipo de usuario (persona sorda / familiar / institución / oyente), y un identificador. Las estadísticas que se recogen son agregadas y no privadas: número de sesiones, cantidad de señas reconocidas, tiempo de uso. Nunca se guarda el contenido de las conversaciones ni video ni audio.

### Pantalla 1.5 — Inicio / Resumen (usuario recurrente)
Lo que ve el usuario que ya pasó bienvenida y cuenta. Es un panel de inicio con accesos rápidos:
- Botón grande "Iniciar conversación" (lleva a Conversar y genera el QR)
- Acceso a Aprender, Biblioteca, Ajustes
- Si tiene cuenta: un resumen breve de su uso (sesiones, señas aprendidas)
- Conversaciones guardadas, si las hay

### Pantalla 2 — Conversar (la principal)
La pantalla central donde ocurre la conversación. Al entrar, ofrece "Iniciar conversación", que genera y muestra el código QR para que el oyente se una. Una vez conectado el oyente, la pantalla muestra:
- Vista de cámara grande con los landmarks dibujados
- La palabra que se está reconociendo con su barra de confianza
- El historial de la conversación: lo que la persona sorda ha dicho (señas traducidas) y lo que el oyente ha respondido (texto que llega de su página web)
- Un botón para decir la frase acumulada y enviarla al oyente
- Indicador de si el oyente sigue conectado
Antes de que el oyente escanee, esta pantalla muestra el QR de forma prominente.

### Pantalla 3 — Aprender / Enseñar señas
Donde el usuario le enseña señas nuevas a la app:
- Capturar muestras de una seña nueva (haciendo la seña frente a la cámara varias veces)
- Ver cuántas muestras se han capturado
- Procesar los keypoints
- Entrenar el modelo
En móvil esto es guiado paso a paso. En escritorio se ve todo junto.

### Pantalla 4 — Biblioteca
Lista de todas las señas que el modelo ya conoce:
- Cada seña con cuántas muestras tiene
- Opción de añadir más ejemplos a una seña existente
- Opción de eliminar una seña

### Pantalla 5 — Ajustes
- Qué cámara usar (si hay varias)
- Idioma y voz del Text-to-Speech
- Umbral de confianza para el reconocimiento
- Velocidad de captura
- Modo claro / oscuro
- Iniciar sesión / cerrar sesión / gestionar cuenta

---

## 6. LAS TRES SUPERFICIES DE LA APP

Ahora hay tres cosas distintas que construir, no dos:

### A) App de la persona sorda — MÓVIL
App móvil instalable. Es la que se lleva a todos lados. Al abrir, lo primero es la pantalla de Conversar. El resto (Aprender, Biblioteca, Ajustes) en una barra inferior de pestañas. Pantallas de una en una, a pantalla completa.

### B) App de la persona sorda — ESCRITORIO
Misma app, más espacio. La pantalla de Conversar muestra a la vez la cámara grande, el QR (al inicio), el historial y los controles. Mejor para entrenar el modelo y gestionar la biblioteca en sesiones largas.

### C) Página web del oyente — NAVEGADOR
Web ligera, no se instala. Se abre al escanear el QR. Solo muestra la conversación y permite responder escribiendo o hablando. Debe verse bien en el teléfono de cualquier persona.

### La regla de adaptabilidad
Las tres superficies se adaptan fluidamente a cualquier tamaño. En pantallas anchas los paneles se acomodan lado a lado; en estrechas se apilan. Nada se corta ni se desborda. Mismo código base adaptándose.

### Decisión de plataforma pendiente
La app de la persona sorda: en escritorio es Tauri (ya está encaminado). En móvil, hay que decidir si es una app nativa real instalable desde tienda, o una web app que se comporta como app. La página del oyente siempre es web.

---

## 7. EL SISTEMA DE CUENTAS — CÓMO SE CONSTRUYE

La autenticación usa Supabase, un servicio externo que ya tiene resuelto el registro seguro, el login y la base de datos. Esto evita construir y asegurar un servidor desde cero.

Lo que hay que tener claro:
- La cuenta de Supabase la crea el equipo de SeñAlerta (es gratis)
- Las claves del proyecto de Supabase se pegan en la configuración de la app
- El backend Python local (localhost:8765) sigue siendo solo para el reconocimiento — no maneja cuentas
- Supabase maneja las cuentas y las estadísticas, en la nube
- Nadie del equipo maneja contraseñas directamente — Supabase las cifra automáticamente

Datos que se recogen (todos no privados y agregados):
- Número de usuarios registrados y su tipo
- Número de sesiones de uso
- Cantidad de señas reconocidas en total
- Tiempo de uso

Datos que NUNCA se recogen:
- El contenido de las conversaciones
- Video o audio de las sesiones
- Cualquier dato que identifique qué dijo una persona

Consideración legal importante: como el público incluye personas sordas y posiblemente menores en instituciones educativas, el manejo de datos debe cumplir la Ley 1581 de protección de datos personales de Colombia. Por eso la cuenta es opcional y solo se recogen datos agregados no identificables.

---

## 8. ARQUITECTURA TÉCNICA RESUMIDA

Cuatro piezas que trabajan juntas:

1. App de la persona sorda (frontend) — React, adaptable a móvil y escritorio. Lo que ve y usa la persona sorda.
2. Backend local de reconocimiento — Python + Flask en localhost:8765, corre en el dispositivo de la persona sorda. Hace MediaPipe, LSTM, captura, entrenamiento. Ya construido y probado.
3. Página web del oyente — web ligera alojada en internet, se abre por QR. Sin cámara ni reconocimiento.
4. Supabase (en la nube) — hace tres cosas: cuentas (opcional), estadísticas de uso, y el canal de tiempo real que conecta la app de la persona sorda con la página del oyente.

Flujo: la app de la persona sorda habla con el backend local para el reconocimiento, y con Supabase para cuentas, estadísticas y la sesión en vivo. La página del oyente solo habla con Supabase Realtime.

---

## 9. ORDEN DE CONSTRUCCIÓN PROPUESTO

1. Definir y aprobar este documento de funcionamiento (estamos aquí)
2. Decidir el diseño visual de cada pantalla (siguiente paso)
3. Construir las pantallas con el diseño aprobado, conectadas al backend local que ya funciona
4. Añadir el sistema de cuentas con Supabase al final, como capa opcional
5. Empaquetar todo como app de escritorio (Tauri) y preparar la versión móvil

---

## DECISIONES TOMADAS (cierre de la fase de funcionamiento)

### Plataforma móvil: app nativa de tienda con reconocimiento local (Camino B)
La app de la persona sorda en móvil es una app nativa instalable desde Play Store / App Store. El reconocimiento de señas corre DENTRO del teléfono, sin internet, usando:
- MediaPipe Tasks para móvil (en lugar de MediaPipe Holistic de Python)
- TensorFlow Lite (en lugar de TensorFlow normal)
- El mismo modelo LSTM ya entrenado, convertido de .keras a .tflite (no se reentrena)

El formato de keypoints es idéntico (1662 valores) y la lógica de reconocimiento es la misma (acumular frames, umbral 0.8, armar frase). Solo se reescribe en el lenguaje de la app móvil.

Tecnología sugerida para la app móvil: React Native o Flutter (comparte código entre Android e iOS).

### Resumen de plataformas
- Escritorio (Windows): app Tauri + backend Python local (YA FUNCIONA, sin cambios)
- Móvil (Android/iOS): app nativa + MediaPipe Tasks + TensorFlow Lite + modelo convertido
- Página del oyente: web ligera por QR (igual en ambos casos)
- Supabase: cuentas, estadísticas, canal de tiempo real del QR

### Pantalla inicial del usuario recurrente
El usuario que ya tiene cuenta y pasó la bienvenida ve primero un INICIO / RESUMEN con accesos rápidos (iniciar conversación, ir a aprender, ver biblioteca), no directo a Conversar.

### Conversaciones
Se pueden guardar si el usuario quiere. Por defecto la conversación es en vivo, pero al terminar la sesión se ofrece guardarla. Las guardadas quedan asociadas a la cuenta (si tiene). Sin cuenta, se pueden guardar localmente en el dispositivo.
