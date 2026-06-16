# ¿Qué es una red LSTM y por qué la usamos para señas?

## Primero: ¿qué es una red neuronal?

Una red neuronal es un programa que aprende de ejemplos en lugar de seguir reglas escritas a mano. Nadie le programa "si la mano sube así, es la seña HOLA". En cambio, le mostramos muchos ejemplos de cada seña y ella sola descubre los patrones que las distinguen.

Es como aprender a montar bicicleta: nadie te da una fórmula, practicas muchas veces hasta que tu cerebro capta el patrón.

## El problema: una seña es MOVIMIENTO, no una foto

Si tomas una sola foto de alguien haciendo una seña, muchas veces no se puede saber qué seña es. Por ejemplo, "hola" y "adiós" pueden verse idénticas en una foto (la mano levantada), porque lo que las diferencia es **cómo se mueve la mano a lo largo del tiempo**.

Una seña es una **secuencia**: una serie de posiciones que ocurren en orden. Como una palabra es una serie de letras en orden — "amor" y "roma" tienen las mismas letras, pero el orden lo cambia todo.

Por eso no nos sirve una red neuronal normal que mira una sola imagen. Necesitamos una que entienda secuencias: que recuerde lo que vio hace un momento para interpretar lo que ve ahora.

## La solución: LSTM, una red con memoria

LSTM significa *Long Short-Term Memory* ("memoria de corto y largo plazo"). Es un tipo de red neuronal diseñada para trabajar con secuencias.

La idea clave: la LSTM procesa la secuencia **frame por frame** (cuadro por cuadro de video), y mientras avanza va guardando un "resumen" interno de lo que ha visto. Ese resumen es su memoria.

- Cuando ve el frame 1, anota algo en su memoria.
- Cuando ve el frame 2, combina lo nuevo con lo que recordaba.
- Y así hasta el frame 15.
- Al final, su memoria contiene la "idea" del movimiento completo, y con eso decide qué seña fue.

Además, la LSTM tiene unas "compuertas" que aprenden **qué recordar y qué olvidar**. Si un frame no aporta nada (la mano quieta), puede ignorarlo. Si un frame es clave (el momento donde la mano gira), lo guarda con fuerza. Por eso es mejor que redes de memoria más simples: no se le "diluye" el recuerdo en secuencias largas.

## En SeñAlerta, concretamente

1. La cámara graba a la persona haciendo la seña.
2. De cada frame extraemos las posiciones del cuerpo, la cara y las manos (los "keypoints" — ver [[flujo-de-datos]]).
3. La LSTM recibe **15 frames seguidos** y los lee en orden, como una mini-película.
4. Al terminar de "ver la película", dice: "esto fue la seña GRACIAS con 95% de seguridad".

Nuestro modelo usa además una variante llamada **Bidirectional LSTM** en su primera capa: lee la secuencia hacia adelante Y hacia atrás. Es como entender mejor una frase leyéndola dos veces, una en cada dirección — capta tanto cómo empieza el movimiento como cómo termina. Los detalles capa por capa están en [[arquitectura-del-modelo]].

## Resumen en una frase

Usamos una LSTM porque las señas son movimientos en el tiempo, y la LSTM es la red neuronal que sabe "ver películas" en lugar de fotos: tiene memoria para entender secuencias.
