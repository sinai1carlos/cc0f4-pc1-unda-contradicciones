# CC0F4 – PC1: Detección de contradicciones normativas (prototipo mínimo)

**Autor:** Carlos Sinai Unda Miguel
**Tema:** Detección automática de contradicciones normativas en textos jurídicos peruanos mediante Retrieval-Augmented Generation e Inferencia de Lenguaje Natural.

## 1. Problema

Dos normas pueden regular el mismo supuesto con consecuencias incompatibles: plazos distintos, una obliga y otra prohíbe, o cambian la vigencia de un permiso. Detectar esto manualmente en corpus grandes es costoso. El proyecto completo de investigación usará RAG para recuperar normas relacionadas y un componente de inferencia de lenguaje natural (NLI) para clasificar la relación entre ellas. Esta práctica **no implementa RAG**: aísla y valida únicamente la unidad de inferencia NLI que alimentará el sistema completo.

## 2. Tarea mínima

Dado un **par de fragmentos normativos** (Norma A, Norma B), clasificar su relación en una de tres categorías:

- `contradiction`: cumplir ambas normas a la vez es imposible.
- `entailment`: una norma implica o es un caso particular de la otra.
- `neutral`: regulan materias distintas y no se afectan entre sí.

Sin recuperación: el par se entrega directamente al modelo, ya emparejado.

## 3. Relación con Semanas 1, 2 y 3

**Semana 1 — Transformer y causalidad.**
Qwen2.5-0.5B-Instruct es un decoder causal: al generar el token de `label`, solo tiene acceso a lo que ya escribió y al contexto que puse antes en el prompt. Por eso Norma A y Norma B se concatenan en un único input — la Norma B "ve" a la Norma A dentro de la atención causal, pero no al revés dentro de esa misma pasada. Si el orden de las normas importara para algún caso (p. ej. una contradicción asimétrica), habría que probarlo explícitamente invirtiendo el orden.

**Semana 2 — logits, softmax y decoding.**
Los logits no son probabilidades; solo tras `softmax` se normalizan a una distribución. Con `decoding=greedy` el modelo toma siempre el argmax de esa distribución: es determinista, y en mis 9 casos convergió a `neutral` en 7-8 de 9, sin variar entre corridas. Con `sampling` (temperature=0.8, top_p=0.9) el token se muestrea de la misma distribución en vez de tomar el máximo — la temperatura no cambia los pesos del modelo, solo cómo se explora esa distribución. Esto explica por qué el caso 1 dio tres etiquetas distintas en 5 corridas de sampling: no es que el modelo "cambie de opinión", es la naturaleza estocástica del muestreo sobre la misma distribución de logits.

**Semana 3 — salida estructurada y validación.**
Distinguí tres niveles: (1) JSON parseable — el texto se puede leer como objeto, incluso envuelto en \`\`\`json; (2) schema-valid — cumple el JSON Schema (enum de label, confidence en [0,1], reason de 1 a 300 caracteres); (3) semánticamente correcto — `label == gold`. Mi Experimento 1 es evidencia directa de que (2) no implica (3): P0 y P1 alcanzan `schema_valid_rate=0.778` pero `accuracy=0.222`, porque el modelo describe correctamente el criterio de contradicción en el campo `reason` (p. ej. "tienen diferentes plazos") y aun así emite `label: neutral`.

## 4. Modelo

`Qwen/Qwen2.5-0.5B-Instruct`, cargado con `torch_dtype="auto"`, ejecutado localmente vía `transformers` (sin API comercial), en CPU dentro de un contenedor Docker con JupyterLab. La elección prioriza reproducibilidad sobre complejidad.

## 5. Datos

9 pares **sintéticos** en estilo normativo peruano (`data/cases.jsonl`): 4 `contradiction`, 2 `entailment`, 3 `neutral`. No citan artículos reales de normas peruanas existentes, para no atribuirles texto inventado. Son adecuados para la tarea porque cada par tiene una etiqueta gold inequívoca y cubren las tres relaciones posibles de forma balanceada.

## 6. Salida estructurada

```json
{"label": "contradiction", "confidence": 0.82, "reason": "..."}
```

Definida en `data/schema.json`: `label` restringido a un enum de 3 valores, `confidence` en [0,1], `reason` de 1 a 300 caracteres, sin campos adicionales (`additionalProperties: false`). La validación se hace con `jsonschema.validate` dentro del propio notebook; una salida puede ser JSON válido y cumplir el schema, y aun así ser semánticamente incorrecta (ver Sección 3 y 13).

## 7. Cómo ejecutar

Todo el prototipo vive en `notebook/PC1_unda.ipynb` — validación, ejecución de experimentos y cálculo de métricas están definidos como funciones dentro del propio notebook, sin depender de archivos `.py` propios.

Abrir `notebook/PC1_unda.ipynb` en JupyterLab y ejecutar las celdas en orden:
1. instala dependencias (`pip install --break-system-packages`),
2. ajusta el directorio de trabajo a la raíz del repo,
3. define el validador (`check`, `extract_json`) contra `data/schema.json`,
4. carga el modelo,
5. corre los 3 experimentos y guarda `results/*.jsonl`,
6. calcula y muestra las métricas (`schema_valid_rate`, `accuracy`, `macro_f1`, estabilidad).

## 8. Prompts utilizados

`prompts/baseline.txt` (P0), `prompts/mejorado.txt` (P1), `prompts/context_C0.txt` (contexto neutral), `prompts/context_C1.txt` (contexto conflictivo).

## 9. Experimento 1 — Prompt

**P0 vs P1**, mismo contexto (C0), mismo decoding (greedy), mismos 9 casos.

**Qué cambió exactamente:** P1 agrega (a) definición explícita de cada label con criterio de decisión, (b) instrucción de que el contexto es información auxiliar y no una instrucción, y (c) formato JSON exacto sin texto adicional. P0 solo pide clasificar y devolver JSON.

| Métrica | P0 | P1 |
|---|---|---|
| schema_valid_rate | 0.778 | 0.778 |
| accuracy | 0.222 | 0.222 |
| macro_f1 | 0.133 | 0.133 |
| distribución de predicciones | neutral: 7, null: 2 | neutral: 7, null: 2 |

**Resultado:** las métricas son idénticas. Mejorar el prompt no cambió ni una sola predicción en estos 9 casos.

**Ejemplo (caso 1, gold=contradiction):** en P1 la razón generada dice *"Ambas normas regulan el mismo supuesto (presentar declaración jurada) pero tienen diferentes plazos, lo cual no produce contradicciones"* — el modelo identifica correctamente el criterio (plazos distintos) que yo mismo definí como señal de contradicción, y aun así predice `neutral`. Esto sugiere que el cuello de botella no es la claridad del prompt sino la capacidad del modelo de 0.5B para aplicar el criterio que ya sabe verbalizar.

**Conclusión (proporcional a la evidencia):** en estos 9 casos, un prompt más detallado con criterios explícitos no mejoró ni empeoró la clasificación. No puedo generalizar que "el prompt no importa" — solo que, para este modelo pequeño y este conjunto de casos, la mejora de prompt no fue la variable que determinó el resultado.

## 10. Experimento 2 — Contexto

**C0 (neutral: "ambas normas son de ámbito nacional y vigentes") vs C1 (conflictivo: "la Norma B rige para un régimen especial distinto")**, mismo prompt P1, mismo decoding (greedy), mismos 9 casos.

| Métrica | C0 | C1 |
|---|---|---|
| schema_valid_rate | 0.778 | 0.889 |
| accuracy | 0.222 | 0.222 |
| macro_f1 | 0.133 | 0.121 |
| distribución de predicciones | neutral: 7, null: 2 | neutral: 8, null: 1 |

**Resultado:** con contexto conflictivo el modelo se vuelve *más* conservador (`neutral` sube de 7 a 8 de 9), no menos — contrario a lo que esperaría si el contexto lo empujara a encontrar más contradicciones.

**Caso donde el contexto afecta el comportamiento (caso 8, gold=neutral):** en C0 la razón describe correctamente los dos temas distintos (elección del alcalde vs. arrendamiento de bienes municipales) y predice `neutral` correctamente. En C1, la razón cambia a *"Ambas normas son contradictorias debido a que la duración de la elección del alcalde y la naturaleza de los contratos de arrendamiento son diferentes"* pero **la etiqueta sigue siendo `neutral`** — el texto de justificación se volvió incoherente con la propia etiqueta emitida (schema-valid=false en este caso, porque el `reason` excede el límite de caracteres). Este es un ejemplo de que el contexto conflictivo perturbó el razonamiento verbalizado sin cambiar la decisión final.

**Conclusión:** en estos 9 casos, el contexto conflictivo no cambió las predicciones (`accuracy` idéntica), pero sí degradó la coherencia interna de algunas justificaciones y bajó ligeramente `macro_f1`. Es un resultado válido: el contexto conflictivo no logró inducir un cambio de etiqueta detectable con este modelo y este tamaño de muestra.

## 11. Experimento 3 — Decoding

**Greedy (D0 = `results/exp1_P1.jsonl`) vs sampling (D1: temperature=0.8, top_p=0.9, top_k=50, seed=42+run, 5 corridas)**, mismo prompt P1, mismo contexto C0, mismos 9 casos, `max_new_tokens=150` fijo.

| Métrica | Greedy (D0) | Sampling (D1, n=45) |
|---|---|---|
| schema_valid_rate | 0.778 | 0.933 |
| accuracy | 0.222 | 0.356 |
| macro_f1 | 0.133 | 0.324 |
| estabilidad media de etiqueta | 1.0 (determinista) | 0.622 |

**Observaciones:**
- Sampling mejora accuracy y macro_f1 respecto de greedy, porque explora otras etiquetas además de `neutral` (`contradiction` aparece en 12 de 45 generaciones, vs. 0 en greedy).
- La estabilidad por caso es baja: el caso 1 y el caso 8 tienen estabilidad 0.4 (la etiqueta mayoritaria solo se repite en 2 de 5 corridas).
- Un `run` (caso 1, run 4) generó una respuesta truncada por `max_new_tokens=150` sin cerrar el JSON, contando como `json_parseable=false`.

**Conclusión:** no se demuestra que sampling sea universalmente mejor — la mejora de accuracy viene junto con una pérdida de reproducibilidad (misma entrada, misma configuración, distinta salida). Esto es exactamente el trade-off que describe la Semana 2: greedy es determinista pero puede quedar atrapado en un óptimo local (siempre `neutral`); sampling explora la distribución completa pero sacrifica estabilidad.

## 12. Resultados

| Experimento | Variable modificada | Variable fija | Métrica | Resultado | Conclusión |
|---|---|---|---|---|---|
| Prompt | P0 a P1 | modelo, casos, contexto C0, decoding greedy | accuracy, macro_f1, schema_valid_rate | Idénticas (0.222 / 0.133 / 0.778) | El prompt no fue la variable determinante para este modelo y estos casos |
| Contexto | C0 a C1 | modelo, prompt P1, casos, decoding greedy | accuracy, macro_f1, schema_valid_rate | accuracy igual (0.222); macro_f1 baja levemente (0.133 a 0.121); schema_valid_rate sube (0.778 a 0.889) | El contexto conflictivo no cambió etiquetas, pero afectó la coherencia de las justificaciones |
| Decoding | greedy a sampling | modelo, prompt P1, contexto C0, casos | accuracy, macro_f1, estabilidad | accuracy sube (0.222 a 0.356); estabilidad media = 0.622 | Sampling mejora la métrica puntual a costa de reproducibilidad |

## 13. Un error o caso fallido

**Caso id=1** (gold=`contradiction`: plazos de 10 días hábiles vs. 30 días calendario), en el Experimento 3 (sampling, 5 corridas):

```text
run 0: neutral
run 1: contradiction 
run 2: contradiction 
run 3: neutral
run 4: JSON truncado — la generación se corta a los 150 tokens sin cerrar la llave de cierre
```

Este caso concentra tres modos de falla distintos y visibles: (1) el modelo predice etiquetas distintas en corridas idénticas salvo la semilla, (2) ni siquiera con sampling llega a mayoría correcta (2 de 5), y (3) una corrida no produce JSON parseable por truncamiento — evidencia de que `max_new_tokens=150` es insuficiente cuando el modelo decide elaborar una justificación larga antes de cerrar el objeto JSON.

## 14. Limitación

Con 9 casos sintéticos, una diferencia de 1 caso equivale a ~11 puntos porcentuales de accuracy — ninguna de las diferencias observadas entre P0/P1 o C0/C1 es lo bastante grande como para descartar que sea ruido de muestra. El prototipo no modela jerarquía normativa (una norma especial vs. una general) ni vigencia temporal (normas derogadas o modificadas), que son centrales para una detección real de contradicciones jurídicas. Un modelo de 0.5B parámetros tiene capacidad de razonamiento limitada en español jurídico, como lo muestra el patrón de "verbalizar el criterio correcto pero fallar la etiqueta" (Sección 9). Los resultados de sampling no son reproducibles bit a bit entre ejecuciones en distinto hardware, aunque la semilla esté fijada, por diferencias de backend numérico.

## 15. Conclusión

En estos 9 casos, ni mejorar el prompt (P0 a P1) ni introducir contexto conflictivo (C0 a C1) cambiaron una sola predicción bajo decoding greedy: el modelo colapsa sistemáticamente a `neutral`, incluso cuando su propio campo `reason` describe correctamente el criterio de contradicción. La única variable que produjo un cambio medible fue el decoding: sampling subió la accuracy de 0.222 a 0.356, pero con una estabilidad de etiqueta de solo 0.622 entre corridas. Esto sugiere que, para esta tarea y este modelo, el cuello de botella no está en la ingeniería de prompt ni en el contexto, sino en la capacidad del modelo de 0.5B para aplicar consistentemente el criterio de decisión al momento de elegir el token de `label`. No puedo generalizar estos hallazgos a modelos más grandes ni a corpus jurídicos reales; son observaciones específicas de esta muestra de 9 casos sintéticos.

## 16. Fuentes

- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.