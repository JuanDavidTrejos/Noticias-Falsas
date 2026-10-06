# Informe de la práctica: análisis lingüístico de noticias

## 1. Creación y caracterización del corpus

### 1.1. Objetivo

Se construyó un corpus balanceado de noticias en español para comparar
características lingüísticas entre noticias verdaderas y falsas. El corpus se
organizó en dos carpetas:

```text
corpus/
├── Falso/
├── Verdad/
└── metadata.csv
```

Los archivos `.txt` son la fuente original del análisis. Cada archivo contiene
el título y el cuerpo de una noticia, conservando el texto crudo, sus
mayúsculas, tildes, signos de puntuación y estructura original. El
preprocesamiento se aplica posteriormente sobre copias en memoria y no
sobrescribe estos archivos.

### 1.2. Fuentes de información

El corpus definitivo combina dos fuentes:

1. **Dataset de Kaggle**: conjunto de noticias políticas en español,
   descargado mediante `kagglehub`. Después de la validación estructural se
   seleccionó una muestra balanceada de 5.000 documentos:
   - 2.500 noticias falsas.
   - 2.500 noticias verdaderas.
2. **Web scraping**:
   - 150 noticias falsas obtenidas de Newtral.
   - 150 noticias verdaderas obtenidas de BBC Mundo.

En total, el corpus está compuesto por **5.300 documentos**, distribuidos de
la siguiente manera:

| Categoría | Dataset | Scraping | Total |
|---|---:|---:|---:|
| Falsas | 2.500 | 150 | 2.650 |
| Verdaderas | 2.500 | 150 | 2.650 |
| **Total** | **5.000** | **300** | **5.300** |

### 1.3. Proceso de construcción

La construcción se realizó mediante dos scripts modulares:

- `src/get_corpus_dataset.py`: descarga, valida, limpia estructuralmente,
  balancea y exporta la muestra del dataset.
- `src/scrape_corpus.py`: consulta las páginas de las fuentes web, extrae
  títulos y cuerpos de noticia y guarda los documentos obtenidos.

Ambos scripts escriben directamente en `corpus/Falso/` y
`corpus/Verdad/`. Para diferenciar las procedencias se utilizan prefijos:

- `ds_falso_####.txt` y `ds_verdad_####.txt` para el dataset.
- `scr_newtral_####.txt` para Newtral.
- `scr_bbc_mundo_####.txt` para BBC Mundo.

El archivo `metadata.csv` relaciona cada documento con su identificador,
etiqueta, fuente, archivo, URL cuando está disponible, título y número de
palabras. La etiqueta utilizada es:

- `0`: noticia falsa.
- `1`: noticia verdadera.

### 1.4. Validación y caracterización

Durante la preparación del dataset se aplicaron las siguientes validaciones:

- Detección de columnas equivalentes a etiqueta, título y texto.
- Conversión de las etiquetas a valores binarios `0/1`.
- Unión del título y el cuerpo en un único campo textual.
- Eliminación de noticias vacías o con menos de 20 palabras.
- Eliminación de duplicados.
- Eliminación de textos idénticos asociados a etiquetas contradictorias.
- Muestreo aleatorio reproducible mediante una semilla fija.
- Balanceo de las dos clases.

**[Insertar aquí la tabla de distribución del corpus por fuente y clase,
obtenida desde `metadata.csv`.]**

**[Insertar aquí la tabla de métricas de longitud del corpus: media,
desviación estándar, mínimo, máximo, cuartiles y suma de palabras por fuente.]**

La caracterización por longitud se calcula sobre el texto crudo. Las
métricas consideradas son:

- **Mean**: promedio de palabras por documento.
- **Std**: desviación estándar de la longitud de los documentos.
- **Min** y **Max**: longitudes mínima y máxima.
- **Q25**, **Q50** y **Q75**: primer cuartil, mediana y tercer cuartil.
- **Sum**: cantidad total de palabras de la fuente.

---

## 2. Preprocesamiento, tokenización, reducción y normalización

El preprocesamiento se implementó en `src/preprocess.py`. Su objetivo es
preparar una representación auxiliar para las fases posteriores, sin perder
el texto original del corpus.

### 2.1. Limpieza aplicada

La función `clean_text` aplica, en orden, las siguientes operaciones:

1. Eliminación de etiquetas HTML.
2. Eliminación de URLs, dominios y enlaces.
3. Eliminación de emoticones ASCII.
4. Eliminación de números enteros y decimales.
5. Conversión de todo el texto a minúsculas.
6. Eliminación de tildes mediante normalización Unicode.
7. Eliminación de signos de puntuación.
8. Conversión de saltos de línea, tabulaciones y retornos de carro en
   espacios.
9. Reducción de espacios consecutivos y eliminación de espacios al inicio y
   al final.

El resultado se almacena en una columna nueva denominada `clean_text`. Los
archivos `.txt` originales no se modifican.

### 2.2. Tokenización

Para el análisis lingüístico se utiliza el tokenizador incluido en el modelo
español de spaCy `es_core_news_sm`. El texto se procesa mediante `nlp.pipe`,
lo que permite procesar los documentos por lotes de forma más eficiente.

La tokenización de spaCy conserva unidades lingüísticas y separa signos,
palabras y otros elementos de acuerdo con sus reglas para español. Para el
conteo POS se excluyen los tokens que son espacios o signos de puntuación.

### 2.3. Reducción y normalización

En esta fase se aplicó una reducción estructural del texto mediante la
eliminación de URLs, números, HTML, emoticones, signos y espacios
innecesarios. También se normalizaron las mayúsculas y las tildes para
facilitar tareas basadas en coincidencia textual.

No se aplicaron todavía stemming, lematización general ni eliminación de
stopwords sobre `clean_text`. La lematización sí se utiliza en el análisis de
frecuencias POS para agrupar formas relacionadas al obtener los lemas más
frecuentes. Esta separación permite conservar el texto natural para POS y NER
y reservar las transformaciones más agresivas para las fases de características
y clasificación.

**[Insertar aquí un ejemplo antes/después de la limpieza, mostrando un
fragmento del texto crudo y su correspondiente `clean_text`.]**

---

## 3. Análisis lingüístico mediante POS Tagging

### 3.1. Metodología

El etiquetado gramatical se realizó con spaCy y el modelo
`es_core_news_sm`. Se analizaron separadamente las noticias verdaderas y
falsas. Para cada token se obtuvo la categoría gramatical (`token.pos_`) y,
para las frecuencias de palabras, el lema (`token.lemma_`).

Las categorías exigidas fueron:

- `NOUN`: sustantivos.
- `VERB`: verbos.
- `ADJ`: adjetivos.
- `ADV`: adverbios.
- `PRON`: pronombres.

La frecuencia absoluta corresponde al número de tokens identificados con una
categoría. La frecuencia relativa se calcula respecto al total de tokens de
las cinco categorías POS analizadas dentro de cada clase. Por esta razón, las
proporciones de cada clase suman aproximadamente 100%; una diferencia mínima
como 100,01% se debe al redondeo a dos decimales.

### 3.2. Resultados comparativos

Los resultados obtenidos fueron:

| Categoría | Verdaderas (cantidad) | Verdaderas (%) | Falsas (cantidad) | Falsas (%) |
|---|---:|---:|---:|---:|
| ADJ | 20.893 | 14,95 | 16.293 | 15,06 |
| ADV | 9.896 | 7,08 | 6.589 | 6,09 |
| NOUN | 63.003 | 45,08 | 51.182 | 47,31 |
| PRON | 14.407 | 10,31 | 9.499 | 8,78 |
| VERB | 31.572 | 22,59 | 24.617 | 22,76 |

**[Insertar aquí la tabla comparativa POS generada en el notebook.]**

**[Insertar aquí el gráfico de barras comparativo de la distribución POS.]**

### 3.3. Interpretación

Los sustantivos son la categoría dominante en ambas clases. Su proporción es
ligeramente mayor en las noticias falsas (47,31%) que en las verdaderas
(45,08%). Esto indica una mayor concentración relativa de nombres de personas,
lugares, instituciones y conceptos en la clase falsa, aunque la diferencia no
permite establecer por sí sola una regla de clasificación.

Los verbos presentan proporciones muy similares: 22,76% en noticias falsas y
22,59% en verdaderas. Los adjetivos también son cercanos, con 15,06% y
14,95%, respectivamente.

Las diferencias más visibles aparecen en adverbios y pronombres. Los
adverbios representan 7,08% en noticias verdaderas frente a 6,09% en falsas,
mientras que los pronombres representan 10,31% y 8,78%, respectivamente.
Estas diferencias pueden estar relacionadas con el estilo periodístico, la
longitud, la fuente o los temas tratados. Por ello, deben analizarse junto con
las métricas de longitud y no interpretarse como evidencia causal.

### 3.4. Palabras más frecuentes

Se obtuvieron los diez lemas más frecuentes para sustantivos, verbos y
adjetivos, separados por clase.

**[Insertar aquí las seis tablas top-10: sustantivos verdaderas, sustantivos
falsas, verbos verdaderas, verbos falsas, adjetivos verdaderas y adjetivos
falsas.]**

La comparación de estos listados permite observar diferencias temáticas y de
vocabulario. Sin embargo, las frecuencias pueden estar influenciadas por la
fuente de origen y por la repetición de temas políticos, por lo que se deben
interpretar como tendencias descriptivas y no como rasgos lingüísticos
universales de la falsedad.

---

## 4. Reconocimiento de Entidades Nombradas (NER)

### 4.1. Metodología

El reconocimiento de entidades se realizó con spaCy y el modelo español
`es_core_news_sm`, procesando los textos crudos del corpus. Se conservó una
fila por cada mención reconocida, incluyendo:

- Identificador del documento (`doc_id`).
- Etiqueta numérica (`label`).
- Clase textual (`clase`).
- Texto de la entidad (`entidad`).
- Tipo de entidad (`tipo`).

Las menciones repetidas no se eliminaron, ya que son necesarias para estudiar
la frecuencia de aparición. Las categorías observadas en el corpus fueron:

- `LOC`: lugares.
- `PER`: personas.
- `ORG`: organizaciones.
- `MISC`: entidades misceláneas.

### 4.2. Distribución general

El modelo reconoció **50.648 menciones de entidades** en **5.286 documentos**.
La distribución por clase fue:

| Clase | Menciones | Entidades únicas | Documentos con entidades |
|---|---:|---:|---:|
| Falsas | 24.207 | 8.025 | 2.646 |
| Verdaderas | 26.441 | 9.994 | 2.640 |

La distribución porcentual por tipo fue:

| Tipo | Falsas (%) | Verdaderas (%) |
|---|---:|---:|
| LOC | 31,57 | 33,15 |
| PER | 25,16 | 25,03 |
| MISC | 23,51 | 25,15 |
| ORG | 19,76 | 16,67 |

**[Insertar aquí la tabla de distribución de entidades por clase y tipo.]**

**[Insertar aquí el gráfico comparativo porcentual de tipos de entidad.]**

Las noticias verdaderas presentan una mayor proporción de lugares y entidades
misceláneas, mientras que las falsas presentan una proporción mayor de
organizaciones. La proporción de personas es prácticamente igual entre las
dos clases.

### 4.3. Entidades más frecuentes

Se calcularon dos tipos de listados:

1. Top-10 de entidades por cada tipo (`LOC`, `PER`, `ORG`, `MISC`) y clase.
2. Top-20 de entidades globales por clase, sin separar por tipo.

**[Insertar aquí las tablas top-10 por tipo para noticias verdaderas y
falsas.]**

**[Insertar aquí la tabla top-20 global de entidades verdaderas.]**

**[Insertar aquí la tabla top-20 global de entidades falsas.]**

**[Insertar aquí el gráfico de las 20 entidades globales más frecuentes en
noticias verdaderas.]**

**[Insertar aquí el gráfico de las 20 entidades globales más frecuentes en
noticias falsas.]**

En las noticias verdaderas aparecen con alta frecuencia entidades como
`PP`, `Gobierno`, `PSOE`, `Venezuela`, `EE.UU.`, `Vox` y `Congreso`. En las
noticias falsas destacan `Ceuta`, `Gobierno`, `Iniciativa vers per
Catalunya`, `España`, `PP`, `BNG`, `Cristina Narbona`, `Congreso` y
`Marruecos`.

Estos resultados sugieren diferencias temáticas: las noticias falsas tienen
una presencia particularmente alta de algunas entidades territoriales y
partidos o coaliciones regionales, mientras que las verdaderas muestran una
distribución más asociada con acontecimientos internacionales y entidades
políticas generales. No obstante, estas diferencias pueden reflejar la
composición de las fuentes y del dataset, no necesariamente una propiedad
intrínseca de toda noticia falsa o verdadera.

### 4.4. Análisis de errores

El corpus no contiene anotaciones manuales de referencia, por lo que no es
posible calcular automáticamente precisión, exhaustividad o F1 del NER. La
evaluación de errores debe hacerse mediante inspección manual del texto
original y de la predicción almacenada en `ner_entities.csv`.

Se identificaron candidatos claros para revisión:

| Caso | Predicción observada | Posible problema |
|---|---|---|
| `Gobierno` | `LOC` | Una institución fue interpretada como lugar. |
| `Qué` | entidad | Falso positivo: palabra funcional reconocida como entidad. |
| `También` | entidad | Falso positivo producido por el contexto o por el modelo. |
| `Fuente de la imagen` | entidad | Fragmento editorial identificado como entidad. |
| `Publicidad` | entidad | Elemento paratextual reconocido como entidad. |

**[Insertar aquí cinco ejemplos completos con `doc_id`, fragmento del texto,
entidad predicha, tipo asignado, tipo esperado y explicación del error.]**

Los errores pueden clasificarse en:

- **Falsos positivos**: secuencias que no son entidades, pero fueron
  detectadas como tales.
- **Errores de categoría**: la entidad fue detectada, pero recibió un tipo
  incorrecto.
- **Entidades omitidas**: una persona, organización o lugar presente en el
  texto no fue reconocido.
- **Errores por contenido editorial**: encabezados, créditos, publicidad,
  pies de imagen o elementos de navegación interpretados como parte de la
  noticia.

En consecuencia, los resultados NER son útiles para caracterizar tendencias
del corpus, pero deben interpretarse considerando las limitaciones del modelo,
la calidad del scraping y la ausencia de una anotación humana de referencia.

---

## 5. Conclusiones parciales

El corpus construido cumple con el balance requerido entre noticias falsas y
verdaderas y conserva la trazabilidad de cada documento mediante
`metadata.csv`. La separación entre texto crudo y texto preprocesado permite
utilizar una representación natural para POS y NER, manteniendo además una
versión normalizada para las fases de extracción de características y
clasificación.

El análisis POS muestra que los sustantivos son la categoría dominante en
ambas clases. Las diferencias relativas más notables aparecen en adverbios y
pronombres, aunque las distribuciones de verbos y adjetivos son muy
semejantes.

El análisis NER muestra que los lugares son el tipo más frecuente en ambas
clases. Las noticias verdaderas tienen mayor proporción de lugares y entidades
misceláneas, mientras que las falsas tienen mayor proporción de organizaciones.
Las entidades concretas más frecuentes revelan diferencias temáticas, pero
también pueden estar condicionadas por las fuentes utilizadas.

Finalmente, la revisión manual es indispensable para valorar los errores del
NER. Los falsos positivos y las confusiones entre categorías muestran que las
frecuencias extraídas deben complementarse con una inspección cualitativa de
los ejemplos del corpus.

