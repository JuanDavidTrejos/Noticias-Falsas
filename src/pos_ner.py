"""Funciones para el análisis POS de noticias en español.

El módulo contiene la lógica de procesamiento y resumen. Las tablas se
devuelven como DataFrames y las figuras como objetos de matplotlib para que
el notebook decida cuándo mostrarlas y dónde guardarlas.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

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
    total_tokens = Counter()
    texts = dataframe[text_column].fillna("").astype(str).tolist()
    labels = dataframe[label_column].astype(int).tolist()

    for doc, label in zip(nlp.pipe(texts, batch_size=batch_size), labels):
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            total_tokens[label] += 1
            if token.pos_ in REQUIRED_POS:
                counts[label][token.pos_] += 1
                token_counts[(label, token.pos_)][token.lemma_.lower()] += 1

    rows = []
    for pos in REQUIRED_POS:
        for label in (1, 0):
            absolute = counts[label][pos]
            rows.append(
                {
                    "POS": pos,
                    "categoria": POS_NAMES[pos],
                    "label": label,
                    "noticia": "Verdaderas" if label == 1 else "Falsas",
                    "frecuencia_absoluta": absolute,
                    "frecuencia_relativa": (
                        absolute / total_tokens[label] * 100
                        if total_tokens[label]
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
    return paths
