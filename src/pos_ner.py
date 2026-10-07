"""Funciones para el análisis POS de noticias en español.

El módulo contiene la lógica de procesamiento y resumen. Las tablas se
devuelven como DataFrames y las figuras como objetos de matplotlib para que
el notebook decida cuándo mostrarlas y dónde guardarlas.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import pandas as pd

REQUIRED_POS = ("NOUN", "VERB", "ADJ", "ADV", "PRON")
POS_NAMES = {
    "NOUN": "Sustantivos",
    "VERB": "Verbos",
    "ADJ": "Adjetivos",
    "ADV": "Adverbios",
    "PRON": "Pronombres",
}


def load_spanish_model(model_name: str = "es_core_news_sm"):
    """Carga el modelo español de spaCy e informa cómo instalarlo si falta."""
    try:
        import spacy
    except ImportError as error:
        raise ImportError(
            "Falta spaCy. Instala las dependencias con: "
            "py -m pip install -r requirements.txt"
        ) from error
    try:
        return spacy.load(model_name)
    except OSError as error:
        raise OSError(
            f"No se encontró el modelo {model_name!r}. Ejecuta: "
            f"py -m spacy download {model_name}"
        ) from error


def analyze_pos(
    dataframe: pd.DataFrame,
    nlp,
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula frecuencias POS y palabras frecuentes por clase.

    Devuelve:
        pos_counts: una fila por clase y etiqueta POS.
        word_counts: frecuencias de tokens por clase y POS.
    """
    required = {text_column, label_column, "doc_id"}
    missing = required - set(dataframe.columns)
    if missing:
        raise KeyError(f"Faltan columnas para POS: {sorted(missing)}")

    counts = {label: Counter() for label in (0, 1)}
    token_counts = {(label, pos): Counter() for label in (0, 1) for pos in REQUIRED_POS}
    texts = dataframe[text_column].fillna("").astype(str).tolist()
    labels = dataframe[label_column].astype(int).tolist()

    for doc, label in zip(nlp.pipe(texts, batch_size=batch_size), labels):
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.pos_ in REQUIRED_POS:
                counts[label][token.pos_] += 1
                token_counts[(label, token.pos_)][token.lemma_.lower()] += 1

    rows = []
    for pos in REQUIRED_POS:
        for label in (1, 0):
            absolute = counts[label][pos]
            analyzed_tokens = sum(counts[label].values())
            rows.append(
                {
                    "POS": pos,
                    "categoria": POS_NAMES[pos],
                    "label": label,
                    "noticia": "Verdaderas" if label == 1 else "Falsas",
                    "frecuencia_absoluta": absolute,
                    "frecuencia_relativa": (
                        absolute / analyzed_tokens * 100
                        if analyzed_tokens
                        else 0.0
                    ),
                }
            )
    pos_counts = pd.DataFrame(rows)
    words = [
        {
            "label": label,
            "noticia": "Verdaderas" if label == 1 else "Falsas",
            "POS": pos,
            "token": token,
            "frecuencia": frequency,
        }
        for (label, pos), counter in token_counts.items()
        for token, frequency in counter.most_common()
    ]
    return pos_counts, pd.DataFrame(words)


def comparative_table(pos_counts: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla con el formato solicitado en el enunciado."""
    table = pos_counts.pivot(
        index="POS",
        columns="noticia",
        values=["frecuencia_absoluta", "frecuencia_relativa"],
    )
    table.columns = [
        f"{news}_{'cantidad' if metric == 'frecuencia_absoluta' else 'porcentaje'}"
        for metric, news in table.columns
    ]
    return table.rename(
        columns={
            "Verdaderas_cantidad": "Verdaderas (cantidad)",
            "Verdaderas_porcentaje": "Verdaderas (%)",
            "Falsas_cantidad": "Falsas (cantidad)",
            "Falsas_porcentaje": "Falsas (%)",
        }
    ).reindex(columns=[
        "Verdaderas (cantidad)",
        "Verdaderas (%)",
        "Falsas (cantidad)",
        "Falsas (%)",
    ]).round(2)


def top_words(
    word_counts: pd.DataFrame,
    label: int,
    pos: str,
    n: int = 10,
) -> pd.DataFrame:
    """Devuelve los n lemas más frecuentes para una clase y POS."""
    return (
        word_counts[
            (word_counts["label"] == label) & (word_counts["POS"] == pos)
        ]
        .sort_values("frecuencia", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def plot_pos_distribution(pos_counts: pd.DataFrame):
    """Crea el gráfico de barras comparativo de frecuencias relativas."""
    import matplotlib.pyplot as plt

    plot_data = pos_counts.pivot(
        index="categoria", columns="noticia", values="frecuencia_relativa"
    ).reindex([POS_NAMES[pos] for pos in REQUIRED_POS])
    figure, axis = plt.subplots(figsize=(10, 5))
    plot_data.plot(kind="bar", ax=axis)
    axis.set_title("Distribución POS por categoría de noticia")
    axis.set_xlabel("Categoría gramatical")
    axis.set_ylabel("Frecuencia relativa (%)")
    axis.legend(title="Clase")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    return figure


def save_pos_results(
    pos_counts: pd.DataFrame,
    word_counts: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Guarda CSV, tabla comparativa y PNG sin mostrar figuras."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pos_counts": output_dir / "pos_frequencies.csv",
        "word_counts": output_dir / "pos_token_frequencies.csv",
        "table": output_dir / "pos_comparative_table.csv",
        "figure": output_dir / "pos_distribution.png",
    }
    pos_counts.to_csv(paths["pos_counts"], index=False, encoding="utf-8")
    word_counts.to_csv(paths["word_counts"], index=False, encoding="utf-8")
    comparative_table(pos_counts).to_csv(paths["table"], encoding="utf-8")
    figure = plot_pos_distribution(pos_counts)
    figure.savefig(paths["figure"], dpi=160, bbox_inches="tight")
    plt.close(figure)
    return paths


# ---------------------------------------------------------------------------
# NER

NER_COLUMNS = ("doc_id", "label", "clase", "entidad", "tipo")
CLASS_NAMES = {0: "Falsas", 1: "Verdaderas"}


def extract_entities(
    dataframe: pd.DataFrame,
    nlp,
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 64,
) -> pd.DataFrame:
    """Extrae las entidades de cada documento en formato tabular.

    La tabla conserva la trazabilidad con ``doc_id`` y la clase original.
    Cada fila representa una mención de entidad; las repeticiones dentro de
    un documento no se eliminan porque son útiles para medir frecuencia.
    """
    required = {text_column, label_column, "doc_id"}
    missing = required - set(dataframe.columns)
    if missing:
        raise KeyError(f"Faltan columnas para NER: {sorted(missing)}")

    texts = dataframe[text_column].fillna("").astype(str).tolist()
    rows = []
    for record, doc in zip(dataframe.itertuples(index=False), nlp.pipe(
        texts, batch_size=batch_size
    )):
        values = record._asdict()
        label = int(values[label_column])
        for entity in doc.ents:
            text = entity.text.strip()
            entity_type = entity.label_
            if text and entity_type:
                rows.append(
                    {
                        "doc_id": values["doc_id"],
                        "label": label,
                        "clase": CLASS_NAMES.get(label, str(label)),
                        "entidad": text,
                        "tipo": entity_type,
                    }
                )
    return pd.DataFrame(rows, columns=NER_COLUMNS)


def summarize_entities(
    entities: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Genera los resúmenes de menciones y entidades únicas por clase/tipo.

    Devuelve ``by_type`` (menciones y documentos), ``top_entities`` (las
    entidades más repetidas por clase y tipo) y ``by_class`` (totales por
    clase). El resultado mantiene tablas incluso cuando no hay entidades.
    """
    required = set(NER_COLUMNS)
    missing = required - set(entities.columns)
    if missing:
        raise KeyError(f"Faltan columnas para resumir NER: {sorted(missing)}")

    if entities.empty:
        by_type = pd.DataFrame(
            columns=["label", "clase", "tipo", "menciones", "documentos"]
        )
        top_entities = pd.DataFrame(
            columns=["label", "clase", "tipo", "entidad", "frecuencia"]
        )
        by_class = pd.DataFrame(columns=["label", "clase", "menciones", "entidades_unicas", "documentos"])
        return {"by_type": by_type, "top_entities": top_entities, "by_class": by_class}

    by_type = (
        entities.groupby(["label", "clase", "tipo"], as_index=False)
        .agg(
            menciones=("entidad", "size"),
            documentos=("doc_id", "nunique"),
        )
        .sort_values(["label", "menciones", "tipo"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    top_entities = (
        entities.groupby(["label", "clase", "tipo", "entidad"], as_index=False)
        .agg(frecuencia=("entidad", "size"))
        .sort_values(
            ["label", "tipo", "frecuencia", "entidad"],
            ascending=[True, True, False, True],
        )
        .groupby(["label", "clase", "tipo"], group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    by_class = (
        entities.groupby(["label", "clase"], as_index=False)
        .agg(
            menciones=("entidad", "size"),
            entidades_unicas=("entidad", "nunique"),
            documentos=("doc_id", "nunique"),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )
    return {
        "by_type": by_type,
        "top_entities": top_entities,
        "by_class": by_class,
    }


def plot_ner_distribution(entities: pd.DataFrame):
    """Crea un gráfico comparativo porcentual por tipo y clase."""
    import matplotlib.pyplot as plt

    summary = entity_type_distribution(entities)
    figure, axis = plt.subplots(figsize=(11, 5))
    if summary.empty:
        axis.text(0.5, 0.5, "No se encontraron entidades", ha="center", va="center")
        axis.set_axis_off()
    else:
        plot_data = summary.pivot_table(
            index="tipo", columns="clase", values="porcentaje", fill_value=0
        )
        plot_data.plot(kind="bar", ax=axis)
        axis.set_title("Entidades nombradas por tipo y clase")
        axis.set_xlabel("Tipo de entidad")
        axis.set_ylabel("Porcentaje de menciones (%)")
        axis.legend(title="Clase")
        axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    return figure


def entity_type_distribution(entities: pd.DataFrame) -> pd.DataFrame:
    """Calcula menciones y porcentajes de cada tipo dentro de cada clase."""
    summary = summarize_entities(entities)["by_type"].copy()
    if summary.empty:
        return summary.assign(porcentaje=pd.Series(dtype=float))
    totals = summary.groupby("label")["menciones"].transform("sum")
    summary["porcentaje"] = summary["menciones"] / totals * 100
    return summary


def top_global_entities(
    entities: pd.DataFrame,
    label: int,
    n: int = 20,
) -> pd.DataFrame:
    """Devuelve las entidades más frecuentes de una clase, sin separar por tipo."""
    required = set(NER_COLUMNS)
    missing = required - set(entities.columns)
    if missing:
        raise KeyError(f"Faltan columnas para resumir NER: {sorted(missing)}")
    return (
        entities[entities["label"] == label]
        .groupby(["label", "clase", "entidad"], as_index=False)
        .agg(frecuencia=("entidad", "size"))
        .sort_values(["frecuencia", "entidad"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )


def plot_top_entities(entities: pd.DataFrame, label: int, n: int = 20):
    """Crea un gráfico horizontal con las entidades más frecuentes de una clase."""
    import matplotlib.pyplot as plt

    data = top_global_entities(entities, label=label, n=n).sort_values("frecuencia")
    figure, axis = plt.subplots(figsize=(10, 7))
    if data.empty:
        axis.text(0.5, 0.5, "No se encontraron entidades", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.barh(data["entidad"], data["frecuencia"])
        axis.set_title(f"20 entidades más frecuentes: {CLASS_NAMES[label]}")
        axis.set_xlabel("Menciones")
        axis.set_ylabel("Entidad")
    figure.tight_layout()
    return figure


def save_ner_results(
    entities: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> dict[str, Path]:
    """Guarda la extracción, resúmenes y gráfico de NER en ``output_dir``."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_entities(entities, top_n=top_n)
    paths = {
        "entities": output_dir / "ner_entities.csv",
        "summary": output_dir / "ner_summary.csv",
        "by_type": output_dir / "ner_by_type.csv",
        "top_entities": output_dir / "ner_top_entities.csv",
        "by_class": output_dir / "ner_by_class.csv",
        "type_distribution": output_dir / "ner_type_distribution.csv",
        "top_global_true": output_dir / "ner_top_global_verdaderas.csv",
        "top_global_false": output_dir / "ner_top_global_falsas.csv",
        "figure": output_dir / "ner_distribution.png",
        "figure_true": output_dir / "ner_top_verdaderas.png",
        "figure_false": output_dir / "ner_top_falsas.png",
    }
    entities.to_csv(paths["entities"], index=False, encoding="utf-8")
    summaries["by_type"].to_csv(paths["summary"], index=False, encoding="utf-8")
    for key in ("by_type", "top_entities", "by_class"):
        summaries[key].to_csv(paths[key], index=False, encoding="utf-8")
    entity_type_distribution(entities).to_csv(
        paths["type_distribution"], index=False, encoding="utf-8"
    )
    top_global_entities(entities, label=1).to_csv(
        paths["top_global_true"], index=False, encoding="utf-8"
    )
    top_global_entities(entities, label=0).to_csv(
        paths["top_global_false"], index=False, encoding="utf-8"
    )
    figure = plot_ner_distribution(entities)
    figure.savefig(paths["figure"], dpi=160, bbox_inches="tight")
    plt.close(figure)
    true_figure = plot_top_entities(entities, label=1)
    true_figure.savefig(paths["figure_true"], dpi=160, bbox_inches="tight")
    plt.close(true_figure)
    false_figure = plot_top_entities(entities, label=0)
    false_figure.savefig(paths["figure_false"], dpi=160, bbox_inches="tight")
    plt.close(false_figure)
    return paths


# Alias explícito para notebooks y código que prefiera el nombre del análisis.
analyze_ner = extract_entities
extract_ner = extract_entities
summarize_ner = summarize_entities
ner_summary = summarize_entities
