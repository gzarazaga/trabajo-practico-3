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
- `testdata.manual.2009.06.14.csv` (~500 registros, opcional): etiquetado a mano, **3 clases** (0/2/4), con ~28% de registros neutros.

Como ningún modelo entrenado sobre el archivo grande puede predecir "neutral", se evaluaron cuatro alternativas: (A) no usar el archivo chico, (B) sacar los registros neutros y usarlo como test externo, (C) usarlo completo con split train/test propio, (D) entrenar un modelo binario y discretizarlo en 3 clases post-hoc.

**Por qué se filtran los neutros:** el archivo grande se etiquetó por *distant supervision* (emoticon), un proceso que nunca produjo un ejemplo neutral — los modelos entrenados ahí no tienen ninguna noción de esa clase en su frontera de decisión. Incluir neutros en la evaluación (o inventar un umbral que los discretice, opción D) es forzar un supuesto no validado: no hay datos de entrenamiento que respalden dónde debería ir ese corte. Filtrarlos no es descartar información porque sí, es reconocer que "neutral" es una categoría que solo existe en el juicio humano del archivo manual, no en la tarea que el modelo efectivamente aprendió.

**Verificación de independencia:** se chequeó solapamiento por texto y por id entre el archivo manual filtrado (359 registros) y el archivo de entrenamiento — 0 coincidencias en ambos casos, confirmando que es un test genuinamente out-of-sample.

**Regla de implementación:** el archivo manual filtrado nunca pasa por `.fit()` de ningún componente (`Word2Vec`, `CountVectorizer`/`TfidfVectorizer`, clasificador) — todo se ajusta solo sobre el split del archivo grande, y el manual se usa exclusivamente en `.transform()`/`.predict()`. Así, filtrar sus neutros no afecta ningún parámetro aprendido, solo qué filas se usan para medir el modelo ya entrenado.

**Decisión adoptada: opción B.** Se filtran los registros neutros (quedan ~359) y se usan exclusivamente como **segundo test set, out-of-sample e independiente** del split del archivo grande, para validar si los modelos generalizan de etiquetas automáticas (por emoticon) a etiquetas puestas por un humano. Se descartaron A (se pierde el único ground truth humano del dataset), C (con ~500 filas el split por clase queda con varianza altísima y no aporta nada entrenar junto a 1.6M) y D (requiere inventar un umbral de "zona neutral" sin datos que lo respalden). Detalle completo en `notebooks/00_lectura_y_discovery.ipynb`.

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

Más allá de lo pedido, se suma un análisis exploratorio pensado como diferenciador (a pedido de la cátedra: "salir de lo típico"): en lugar de solo clasificar sentimiento, se busca **detectar eventos reales a partir de picos temáticos en el tiempo**.

- Entrenar **BERTopic** una vez sobre el corpus (o una muestra representativa) para obtener los tópicos.
- Usar `topics_over_time()` para bucketizar por la columna `date` sin re-clusterizar.
- Buscar picos de volumen por tópico/día y contrastarlos contra eventos conocidos del rango de fechas del dataset (abril–junio 2009): protestas post-electorales en Irán (mediados de junio) y la muerte de Michael Jackson (25/6/2009, límite superior del dataset).
- Cruzar el pico temático con la polaridad promedio de sentimiento de esos tweets ese día, conectando este análisis con el resto del TP en lugar de dejarlo aislado.
- BERTopic usa cosine similarity internamente (c-TF-IDF y clustering), por lo que también aporta a la métrica obligatoria.

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
    └── 05_topicos_temporales.ipynb      # Diferenciador: BERTopic + picos por día vs. eventos reales
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

## Referencias

- Consigna oficial: `Instrucciones/Consignas_proyecto_NLP.ipynb`
- Dataset: [Sentiment140](https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit)
- Notebooks de referencia de clase: `nlp_sentiment.ipynb`, `Embeddings.ipynb`, `Word Embedding - caso practico.ipynb`
- Metodología original: Go, A., Bhayani, R. & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*, Stanford University — documenta que el archivo de entrenamiento se etiquetó por distant supervision (emoticones) y que `testdata.manual...csv` es el test set anotado a mano por los autores.
