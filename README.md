# TP3 — Análisis de Sentimiento en Tweets con Word2Vec

**Diplomatura en Ciencia de Datos**
**Fecha de entrega:** 16/07/2026

---

## Descripción del problema

Análisis de sentimiento sobre el dataset **Sentiment140**, compuesto por aproximadamente **1.6 millones de tweets** etiquetados con polaridad:

- `0` → Negativo
- `2` → Neutral
- `4` → Positivo

El objetivo es entrenar y comparar múltiples enfoques de clasificación de sentimiento, incorporando embeddings semánticos entrenados sobre el corpus de tweets, y validar el modelo resultante contra un baseline pre-entrenado.

---

## Enfoque propuesto

Se combinan **tres enfoques progresivos** que permiten comparar representaciones de texto de distinta complejidad:

### Enfoque 1 — Baseline pre-entrenado (TextBlob)
- Aplicar TextBlob directamente sobre los tweets sin entrenamiento propio.
- Sirve como cota inferior de referencia.

### Enfoque 2 — Modelos clásicos con representaciones dispersas
- **BoW + Naive Bayes**: representación por frecuencia de palabras.
- **TF-IDF + Logistic Regression**: representación ponderada por relevancia del término en el corpus.
- Ambos modelos entrenados sobre el archivo de entrenamiento completo (1.6M registros).

### Enfoque 3 — Word2Vec + Logistic Regression (enfoque principal)
- Entrenar un modelo **Word2Vec** (gensim) sobre los 1.6M tweets.
- Representar cada tweet como el **promedio de los embeddings** de sus palabras.
- Entrenar un clasificador (Logistic Regression) sobre esas representaciones densas.
- Explorar el espacio semántico aprendido: similitudes, analogías y visualización UMAP.

---

## Decisión sobre el dataset de test manual

El corpus se distribuye en dos archivos con esquemas de etiquetas distintos:

- `training.1600000.processed.noemoticon.csv` (~1.6M, obligatorio): etiquetado automático por emoticon, **2 clases** (0/4).
- `testdata.manual.2009.06.14.csv` (~500 registros, opcional): etiquetado con una metodología distinta a la del archivo grande, **3 clases** (0/2/4), con ~28% de registros neutros.

Como ningún modelo entrenado sobre el archivo grande puede predecir "neutral", se evaluaron cuatro alternativas: (A) no usar el archivo chico, (B) sacar los registros neutros y usarlo como test externo, (C) usarlo completo con split train/test propio, (D) entrenar un modelo binario y discretizarlo en 3 clases post-hoc.

**Por qué se filtran los neutros:** el archivo grande se etiquetó por *distant supervision* (emoticon), un proceso que nunca produjo un ejemplo neutral — los modelos entrenados ahí no tienen ninguna noción de esa clase en su frontera de decisión. Incluir neutros en la evaluación (o inventar un umbral que los discretice, opción D) es forzar un supuesto no validado: no hay datos de entrenamiento que respalden dónde debería ir ese corte. Filtrarlos no es descartar información porque sí, es reconocer que "neutral" es una categoría que no existe en la tarea que el modelo efectivamente aprendió.

**¿Es realmente un etiquetado manual/humano?** Es un punto que vale la pena no dar por sentado. Verificado contra una fuente independiente (Saif, Fernandez, He & Alani, 2014 — ver Referencias), el archivo chico coincide exactamente en composición (177 negativos, 182 positivos, 139 neutros) y en metodología de recolección (tweets buscados por query de producto/marca/persona, consistente con los 81 valores distintos de `query` que encontramos, contra 1 solo en el archivo grande) con lo que la literatura llama "STS-Test". Está bien corroborado que usa un proceso de etiquetado **distinto** al heurístico automático del archivo grande. Lo que **no** está documentado ni por los autores originales ni por esta fuente es cuántos anotadores lo etiquetaron ni el acuerdo entre ellos — a diferencia de otros datasets de la literatura (STS-Gold, SemEval) donde sí se reporta ese detalle. Por eso en el resto del TP se lo trata como "test manual"/"test externo" sin asumir un rigor de anotación que no está probado.

**Verificación de independencia:** se chequeó solapamiento por texto y por id entre el archivo manual filtrado (359 registros) y el archivo de entrenamiento — 0 coincidencias en ambos casos, confirmando que es un test genuinamente out-of-sample.

**Regla de implementación:** el archivo manual filtrado nunca pasa por `.fit()` de ningún componente (`Word2Vec`, `CountVectorizer`/`TfidfVectorizer`, clasificador) — todo se ajusta solo sobre el split del archivo grande, y el manual se usa exclusivamente en `.transform()`/`.predict()`. Así, filtrar sus neutros no afecta ningún parámetro aprendido, solo qué filas se usan para medir el modelo ya entrenado.

**Limitación de tamaño de muestra:** con n=359, cualquier accuracy medida ahí tiene un intervalo de confianza (95%) de aproximadamente ±4 puntos porcentuales. Esto es suficiente para comparar contra un baseline claramente peor (TextBlob), pero no para rankear con precisión entre los tres modelos entrenados — sus intervalos se solapan entre sí. Detalle en `notebooks/04_clasificacion_w2v.ipynb`, sección 6.1.

**Decisión adoptada: opción B.** Se filtran los registros neutros (quedan ~359) y se usan exclusivamente como **segundo test set, out-of-sample e independiente** del split del archivo grande, para chequear si los modelos generalizan más allá de las etiquetas automáticas por emoticon del archivo grande — no como un ranking preciso entre modelos. Se descartaron A (se pierde el único conjunto con un esquema de etiquetado distinto al heurístico automático), C (con ~500 filas el split por clase queda con varianza altísima y no aporta nada entrenar junto a 1.6M) y D (requiere inventar un umbral de "zona neutral" sin datos que lo respalden). Detalle completo en `notebooks/00_lectura_y_discovery.ipynb`.

---

## Datos versionados en el repositorio

`data/raw/` y `data/processed/*.csv` están en `.gitignore` — se regeneran corriendo `00_lectura_y_discovery.ipynb` y `01_preprocesamiento.ipynb` respectivamente. El principal motivo es `train_processed.csv` (~230MB), que supera cómodamente el límite práctico de GitHub.

**Nota pendiente (no decidida todavía):** se evaluó comprimir `train_processed.csv` para poder versionarlo. Mediciones sobre el archivo real:

| Método | Tamaño resultante | Tiempo |
|---|---|---|
| gzip -9 | 78,3 MB | rápido |
| xz -6 | 61,5 MB | ~2 min |
| zstd -19 | 59,8 MB | ~1,3 min |

Los tres entran bajo el límite duro de 100MB de GitHub, pero rondan la zona donde el clone del repo empieza a pesar (~60-80MB) y GitHub ya avisa de "archivo grande". Alternativa a considerar más adelante si se quiere que el repo sea autocontenido sin correr el pipeline: **Git LFS** en vez de comprimir a mano. Por ahora se mantiene sin publicar y se regenera localmente.

---

## Pipeline detallado

### 1. Preprocesamiento específico para tweets
- Eliminar URLs (`http://...`)
- Eliminar menciones (`@usuario`)
- Eliminar hashtags (`#palabra`) — o conservar solo el texto sin `#`
- Normalizar emojis o eliminarlos
- Minúsculas, tokenización con NLTK

### 2. Entrenamiento Word2Vec
- Librería: `gensim.models.Word2Vec`
- Hiperparámetros a explorar: `vector_size` (100–300), `window` (4 y 8), `min_count`, `negative`
- Se comparan ventana chica (semántica) vs. ventana grande (tópicos), siguiendo el enfoque del notebook de clase

### 3. Representación de documentos
- Cada tweet se representa como el promedio de los vectores de sus tokens presentes en el vocabulario del modelo
- Tweets sin ningún token conocido se descartan o se representan con vector cero

### 4. Clasificación y evaluación
- Entrenar LR sobre embeddings promedio
- Evaluar con `classification_report`, matriz de confusión y curva ROC
- Comparar los tres enfoques en una tabla resumen de métricas (Accuracy, F1-score, AUC)

### 5. Análisis semántico del espacio Word2Vec
- **Similitud coseno** entre palabras de sentimiento opuesto: `happy` vs `sad`, `good` vs `bad`, `love` vs `hate`
- **Palabras más cercanas** a términos de sentimiento representativos
- **Analogías**: explorar operaciones vectoriales sobre vocabulario de Twitter
- **UMAP**: visualización 2D de clusters semánticos (emociones, jerga, expresiones)

---

## Métrica obligatoria

**Similitud coseno**, aplicada en dos contextos:
1. En el análisis semántico del modelo Word2Vec: comparación de palabras por proximidad en el espacio de embeddings.
2. Como métrica implícita en la representación de documentos: los embeddings promedio son comparados vía distancia coseno en el espacio vectorial.

---

## Diferenciador — Análisis temporal de tópicos (BERTopic)

Más allá de lo pedido, se suma un análisis exploratorio pensado como diferenciador (a pedido de la cátedra: "salir de lo típico"): en lugar de solo clasificar sentimiento, se busca **detectar eventos reales a partir de picos temáticos en el tiempo**. Implementado en `notebooks/05_topicos_temporales.ipynb`.

- Entrenar **BERTopic** sobre una muestra representativa del corpus (80.000 tweets — embeber 1.6M en CPU es inviable en tiempo razonable) para obtener los tópicos.
- Usar `topics_over_time()` para bucketizar por la columna `date` sin re-clusterizar.
- Buscar picos de volumen por tópico/día y contrastarlos contra eventos conocidos del rango de fechas del dataset (abril–junio 2009).
- Cruzar el pico temático con la polaridad promedio de sentimiento de esos tweets, conectando este análisis con el resto del TP en lugar de dejarlo aislado.
- BERTopic usa cosine similarity internamente (c-TF-IDF y UMAP), por lo que también aporta a la métrica obligatoria.

**Aclaración importante:** los tópicos dominantes por volumen del corpus **no son políticos** — son charla cotidiana de Twitter (sueño/cama, agradecimientos, mascotas, clima, cumpleaños, el propio Twitter). Irán no es "el tópico principal" del dataset; es el caso de éxito elegido entre los eventos candidatos para poner a prueba la metodología de detección de eventos por picos de volumen.

**Resultado:** de los dos eventos puntuales buscados específicamente por caer dentro del rango de fechas del dataset, solo uno dejó una huella detectable como tópico propio. El tópico de la elección/protestas en Irán es prácticamente inexistente antes del 12/6/2009 (fecha de la elección) y explota el 15/6, con ~86% de tweets negativos (vs. ~50% esperado por el balance del dataset) — un pico real y verificable. La hipótesis inicial sobre la muerte de Michael Jackson (25/6/2009, límite superior del dataset) **no se confirmó**: no llegó a formar un tópico propio (solo menciones sueltas y anteriores al hecho) porque la recolección del dataset corta a las 10:28 (PDT) de ese día, ~4 horas antes de que se anunciara su muerte (~14:26 PDT) — se documenta como hallazgo honesto en vez de forzar una conclusión que los datos no respaldan.

---

## Estructura del repositorio

```
TP3/
├── README.md                        # Este archivo
├── Instrucciones/
│   └── Consignas_proyecto_NLP.ipynb
└── notebooks/
    ├── 00_lectura_y_discovery.ipynb     # Exploración de los dos archivos raw + decisión sobre el test manual
    ├── 01_preprocesamiento.ipynb        # Limpieza y tokenización de tweets
    ├── 02_modelos_clasicos.ipynb        # TextBlob, BoW+NB, TF-IDF+LR
    ├── 03_word2vec.ipynb                # Entrenamiento Word2Vec y análisis semántico
    ├── 04_clasificacion_w2v.ipynb       # Clasificador sobre embeddings + comparación final
    ├── 05_topicos_temporales.ipynb      # Diferenciador: BERTopic + picos por día vs. eventos reales
    └── 06_conclusiones.ipynb            # Conclusiones generales: comparación final de modelos + hallazgos de tópicos
```

> Alternativa: notebook único `TP3_zarazaga.ipynb` con todas las secciones si se prefiere una entrega integrada.

---

## Stack tecnológico

| Herramienta | Uso |
|---|---|
| `pandas`, `numpy` | Manipulación de datos |
| `nltk` | Tokenización, stopwords |
| `re` | Preprocesamiento de texto (regex para tweets) |
| `gensim` | Entrenamiento de Word2Vec |
| `scikit-learn` | BoW, TF-IDF, Naive Bayes, Logistic Regression, métricas |
| `textblob` | Baseline pre-entrenado |
| `umap-learn` | Visualización 2D del espacio de embeddings |
| `bertopic` | Diferenciador: modelado de tópicos y evolución temporal |
| `matplotlib`, `seaborn` | Gráficos y comparativas |

---

## Decisiones de diseño

- **Por qué Word2Vec propio en lugar de spaCy preentrenado**: Los embeddings de spaCy están entrenados sobre texto formal (web, Wikipedia). Twitter tiene vocabulario específico (slang, abreviaciones, emojis) que un modelo entrenado sobre el corpus propio captura mejor.
- **Por qué ventana chica y ventana grande**: Siguiendo el análisis del notebook de clase, se espera que la ventana chica capture relaciones semánticas (sinónimos de sentimiento) y la ventana grande relaciones temáticas (contextos en los que aparecen).
- **Por qué embedding promedio**: Es el enfoque más simple para pasar de token-level a document-level y permite mantener el clasificador como una caja blanca (LR) con coeficientes interpretables.

---

## Conclusiones generales

Desarrollo completo en `notebooks/06_conclusiones.ipynb`. Resumen:

### Sobre los modelos

| Modelo | Acc (val) | F1 (val) | AUC (val) | Acc (manual) | IC 95% (manual) |
|---|---|---|---|---|---|
| TextBlob (baseline) | 0,612 | 0,699 | 0,688 | 0,688 | [0,64 – 0,74] |
| BoW + Naive Bayes | 0,774 | 0,773 | 0,848 | 0,811 | [0,77 – 0,85] |
| TF-IDF + Logistic Regression | 0,787 | 0,790 | 0,867 | 0,797 | [0,76 – 0,84] |
| Word2Vec + Logistic Regression | 0,771 | 0,771 | 0,851 | 0,816 | [0,78 – 0,86] |

- TextBlob queda último en las tres métricas y en ambos conjuntos — cumple su rol de cota inferior. La diferencia contra los tres modelos entrenados es estadísticamente sólida incluso en el test manual (n=359): su intervalo de confianza no se solapa con el de ninguno.
- TF-IDF+LR lidera en validación con margen razonable.
- **En el test manual, los tres modelos entrenados quedan estadísticamente empatados**: Word2Vec+LR tiene el punto más alto (0,816), pero con n=359 el margen de error ronda ±4pp y los tres intervalos se solapan casi por completo — no se puede afirmar cuál generaliza mejor con esta muestra.
- Lo que sí se sostiene: ningún modelo entrenado se derrumba en el test manual — los tres generalizan más allá del heurístico de etiquetado por emoticon del archivo grande.
- No hay un único "mejor modelo" sin contexto: en validación gana TF-IDF+LR; en el test manual, los tres entrenados están empatados entre sí y todos superan claramente a TextBlob.

### Sobre los tópicos (diferenciador)

- **Los tópicos dominantes por volumen son charla cotidiana**, no política: sueño/cama, agradecimientos, mascotas, clima, cumpleaños, el propio Twitter. Irán **no es "el tópico principal"** del dataset.
- **Confirmado**: de los eventos puntuales buscados específicamente, el tópico de la elección/protestas en Irán es prácticamente inexistente antes del 12/6/2009 y explota el 15/6, con ~86% de sentimiento negativo (vs. ~50% esperado) — validación externa de que BERTopic detecta eventos reales, no ruido.
- **Corregido, no forzado**: la hipótesis sobre Michael Jackson no se sostuvo — no llegó a formar un tópico propio (la recolección del dataset corta ~4hs antes del anuncio de su muerte) — se documentó como hallazgo honesto en vez de sostener una conclusión no respaldada por los datos.

### Métrica obligatoria (cosine similarity), tres contextos

Nivel palabra (Word2Vec, `03`), nivel documento (embeddings promedio, `04`), y modelado de tópicos (BERTopic internamente, `05`).

---

## Referencias

- Consigna oficial: `Instrucciones/Consignas_proyecto_NLP.ipynb`
- Dataset: [Sentiment140](https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit)
- Notebooks de referencia de clase: `nlp_sentiment.ipynb`, `Embeddings.ipynb`, `Word Embedding - caso practico.ipynb`
- Dataset original: Go, A., Bhayani, R. & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*, Stanford University — paper que introduce Sentiment140. No se verificó su texto original en este TP; la caracterización del archivo de entrenamiento (etiquetado por distant supervision/emoticones) surge de la propia estructura de los datos (ver `00_lectura_y_discovery.ipynb`).
- **Verificación independiente del test manual** (sí leída y contrastada dato por dato en este TP): Saif, H., Fernandez, M., He, Y. & Alani, H. (2014). *"Evaluation Datasets for Twitter Sentiment Analysis: A survey and a new dataset, the STS-Gold"*, CEUR Workshop Proceedings Vol-1096. Describe el archivo chico ("STS-Test") con 177 negativos/182 positivos/139 neutros — coincide exactamente con nuestros propios números — y confirma que usa un proceso de etiquetado distinto al heurístico automático del archivo grande. El mismo paper señala como limitación conocida que Go et al. no documentaron cuántos anotadores participaron ni el acuerdo entre ellos.
