# Desafío 5S · Pedagogía V4

## Objetivo

Reducir carga de lectura sin perder aleatoriedad ni capacidad de evaluación entre 57 colaboradores.

## Estructura de cada evaluación nueva

- 15 situaciones totales.
- 5 bloques, uno por cada principio 5S.
- Antes de cada bloque aparece una micro-pantalla **Qué esperamos**.
- Cada bloque contiene 3 situaciones elegidas al azar:
  - 2 preguntas situacionales de un banco de 4 por S.
  - 1 desafío visual de un banco de 5–6 fotos por S.
- Dentro de cada bloque las 3 situaciones se mezclan.
- Cada pregunta muestra 3 alternativas tomadas del banco de 4, incluyendo siempre la correcta y 2 distractores aleatorios.
- Las 3 alternativas se mezclan.
- Cada desafío visual muestra 3 afirmaciones tomadas del banco de 5:
  - 2 verdaderas.
  - 1 falsa.
- Las afirmaciones se mezclan.

## Banco activo

### Situacionales

- S1: 4
- S2: 4
- S3: 4
- S4: 4
- S5: 4

Total: 20.

### Visuales

- S1: 5
- S2: 6
- S3: 6
- S4: 6
- S5: 5

Total: 28.

No se eliminaron preguntas ni fotografías.

## Puntuación

### Decisiones

- 15 decisiones máximas.
- 1 punto por decisión correcta.

### Observación visual

- 5 fotos.
- Cada foto tiene 2 afirmaciones verdaderas visibles.
- Máximo: 2 puntos por foto, 10 puntos totales.
- Acierto marcado: +1.
- Afirmación falsa marcada: -1.
- Piso por foto: 0.

### Índice 5S

`Índice 5S = 50% decisiones + 50% observación visual`

Clasificación:

- **AFIANZADO**: índice >= 80% y al menos una decisión correcta en cada S.
- **REQUIERE_REFUERZO**: índice >= 60%.
- **REQUIERE_REEVALUACION**: índice < 60%.

## Aleatoriedad anti-copia

Antes de considerar el orden de preguntas, alternativas y afirmaciones, las combinaciones posibles de selección de contenido superan los 40 millones. El orden interno y la mezcla de alternativas multiplican ampliamente ese número.

## Compatibilidad

Las evaluaciones creadas antes de V4 conservan sus asignaciones originales. Las evaluaciones nuevas usan la estructura V4. Las funciones de respuesta admiten ambos formatos.

## Supabase

Migraciones aplicadas en producción:

- `desafio5s_pedagogy_content_v4`
- `desafio5s_pedagogy_logic_v4`

Funciones actualizadas:

- `desafio5s_asignar_preguntas`
- `desafio5s_pregunta`
- `desafio5s_responder_v2`
- `desafio5s_revision`

Se agregaron campos de presentación breve en `desafio5s_preguntas`, manteniendo los textos originales como respaldo.
