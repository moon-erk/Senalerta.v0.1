# FLUJO DE TRABAJO — SeñAlerta
## Obsidian (memoria) → Stitch (diseño) → Claude Code (construcción)

Cada herramienta tiene un rol claro y no se pisan. Claude Code es el motor que escribe el código; las otras dos lo alimentan y lo recuerdan.

---

## LOS ROLES — QUIÉN HACE QUÉ

| Herramienta | Rol | Qué NO hace |
|---|---|---|
| Obsidian | Memoria del proyecto: decisiones, docs, diagramas | No programa |
| Stitch | Diseña visualmente las pantallas | No programa la app real |
| Claude Code | Construye la app de verdad | (es el único que toca el código) |
| Antigravity | Opcional, solo para probar en Chrome | No tocar el mismo código que Claude Code |

Regla de oro: solo Claude Code edita el código del proyecto. Si usas Antigravity, que sea en una copia separada solo para pruebas, nunca sobre los mismos archivos.

---

## PARTE 1 — OBSIDIAN COMO MEMORIA

### Cómo organizar el vault (la carpeta de notas)
Crea un vault de Obsidian llamado "SeñAlerta" con esta estructura:

```
SeñAlerta-vault/
├── 00-INDICE.md            ← mapa de todo, enlaces a las demás notas
├── 01-arquitectura.md      ← el documento de arquitectura funcional
├── 02-documento-maestro.md ← todas las decisiones de backend/frontend/UX
├── 03-decisiones.md        ← bitácora: cada decisión con fecha y por qué
├── 04-diagramas/           ← los SVG de flujo
├── 05-prompts/             ← los prompts que le das a Claude Code, guardados
├── 06-progreso.md          ← qué se ha hecho y qué falta (se actualiza siempre)
└── 07-aprendizajes.md      ← errores resueltos, trucos, cosas que funcionaron
```

### Cómo usar Obsidian día a día
- Cada vez que tomas una decisión importante, la anotas en `03-decisiones.md` con la fecha. Así nunca pierdes el "por qué" de algo.
- Cuando Claude Code termina algo, actualizas `06-progreso.md`.
- Cuando resuelves un error difícil, lo anotas en `07-aprendizajes.md` para no repetirlo.
- El `00-INDICE.md` enlaza todo con enlaces internos de Obsidian (`[[01-arquitectura]]`) para navegar rápido.

### El truco para conectar Obsidian con Claude Code
Obsidian guarda archivos `.md` normales. Claude Code lee `.md`. Entonces:

Opción A (recomendada): pon el vault de Obsidian DENTRO de la carpeta del proyecto, en una subcarpeta `/docs`. Así Claude Code puede leer todas tus notas directamente, y tú las editas con Obsidian. Una sola fuente de verdad.

```
proyecto-senalerta/
├── CLAUDE.md
├── docs/              ← este es tu vault de Obsidian
│   ├── 00-INDICE.md
│   ├── 01-arquitectura.md
│   └── ...
├── python/
├── src/
└── ...
```

En tu `CLAUDE.md`, añade una línea que diga: "La documentación y decisiones del proyecto están en /docs. Consúltala cuando necesites contexto sobre arquitectura o decisiones." Así Claude Code sabe que ahí está la memoria.

Opción B: mantienes el vault aparte y copias los documentos clave al proyecto cuando los necesitas. Más manual, menos recomendado.

---

## PARTE 2 — STITCH PARA DISEÑAR LAS PANTALLAS

### Qué es y para qué lo usas
Stitch genera diseños visuales de interfaz a partir de descripciones de texto o imágenes de referencia. Lo usas ANTES de que Claude Code programe cada pantalla, para tener una referencia visual clara.

### Cómo usarlo para SeñAlerta
Para cada una de las 7 pantallas, le das a Stitch:
1. El manual de marca (colores turquesa #18B7B0, coral #F26B4A, blanco; la abeja)
2. Una descripción de la pantalla (la tienes en el documento maestro)
3. Las reglas de UX: iconos grandes + texto, accesibilidad fuerte, equilibrio limpio

Ejemplo de lo que le dirías a Stitch para la pantalla Conversar:
"Diseña una pantalla de app llamada Conversar para una app de traducción de lengua de señas. Colores: turquesa #18B7B0 como primario, naranja coral #F26B4A como acento, fondo blanco. Debe tener: una vista de cámara grande a la izquierda, un panel de historial de conversación a la derecha, una barra de confianza debajo de la cámara, un botón grande para iniciar conversación que muestra un código QR. Iconos grandes con texto siempre visible. Accesible, contraste alto, limpio pero con información visible."

### Qué haces con el resultado de Stitch
Stitch te da el diseño (imágenes y a veces código de referencia). Tú:
1. Guardas las imágenes del diseño en el vault de Obsidian (`04-diagramas/` o una carpeta `disenos/`)
2. Se las pasas a Claude Code como referencia visual cuando le pidas construir esa pantalla

Importante: el código que genera Stitch suele ser una base, no el producto final. Claude Code lo adapta a tu proyecto real, conecta el backend, y aplica la adaptabilidad. No copies el código de Stitch directamente al proyecto sin que Claude Code lo integre.

---

## PARTE 3 — CLAUDE CODE COMO MOTOR

### Cómo le llega todo
Claude Code recibe:
- El contexto y las decisiones desde `/docs` (tu Obsidian)
- La referencia visual desde los diseños de Stitch
- Las instrucciones tuyas en cada prompt

### El ciclo de trabajo con Claude Code
1. Lees el progreso en `06-progreso.md` de Obsidian para saber dónde vas
2. Diseñas la pantalla en Stitch (si es nueva)
3. Le pasas a Claude Code el prompt + el diseño de referencia
4. Claude Code construye y prueba
5. Verificas el resultado
6. Actualizas `06-progreso.md` y `07-aprendizajes.md` en Obsidian
7. Repites con la siguiente pantalla o tarea

---

## EL FLUJO COMPLETO EN UNA IMAGEN

```
   [Obsidian]  ← guardas decisiones, docs, progreso
       │  da contexto
       ▼
   [Stitch]    ← diseñas cómo se ve cada pantalla
       │  da referencia visual
       ▼
   [Claude Code] ← construye la app de verdad, conecta backend
       │  lo aprendido vuelve a
       ▼
   [Obsidian]  ← actualizas progreso y aprendizajes
       
   [Antigravity] ← opcional, aparte, solo para probar en Chrome
```

---

## RECOMENDACIONES FINALES

- No uses Claude Code y Antigravity sobre el mismo código a la vez. Elige Claude Code como motor (ya lo tienes andando) y deja Antigravity solo para experimentos separados si quieres.
- Obsidian es tu seguro contra perder el hilo: si un día vuelves y no recuerdas dónde ibas, `06-progreso.md` te lo dice.
- Stitch acelera el diseño, pero el diseño final lo pule Claude Code dentro del proyecto real.
- Mantén el vault de Obsidian dentro de `/docs` del proyecto para que sea una sola fuente de verdad que Claude Code también lee.
- Cada sesión de Claude Code: empieza pidiéndole que lea CLAUDE.md y /docs/06-progreso.md para ponerse al día.
