"""Convierte un dataset etiquetado en archivos TXT y metadata.csv.

Ejemplo:
    py src/get_corpus_dataset.py --input data/noticias.csv

El CSV debe contener una columna de texto y otra de etiqueta. Las columnas
se pueden cambiar con --text-column y --label-column.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "corpusDataset"


def normalize_label(value: object) -> int:
    """Convierte etiquetas comunes a 0=falsa y 1=verdadera."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)

    normalized = str(value).strip().lower()
    if normalized in {"0", "fake", "falso", "false", "fake news"}:
        return 0
    if normalized in {"1", "real", "verdadero", "verdadera", "true", "real news"}:
        return 1
    raise ValueError(
        f"Etiqueta no reconocida: {value!r}. Usa 0/1, falso/verdadero o fake/real."
    )


def clean_filename(value: object) -> str:
    """Genera un fragmento seguro para usar dentro de un nombre de archivo."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_") or "documento"


def build_doc_id(label: int, sequence: int) -> str:
    """Construye IDs consistentes con la clase del documento."""
    prefix = "ds_falso" if label == 0 else "ds_verdad"
    return f"{prefix}_{sequence:04d}"


def write_dataset_corpus(
    dataframe: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    text_column: str = "text",
    label_column: str = "label",
    source_name: str = "dataset",
) -> pd.DataFrame:
    """Guarda los textos y devuelve el DataFrame de metadatos generado."""
    if text_column not in dataframe.columns or label_column not in dataframe.columns:
        raise KeyError(
            f"El dataset debe incluir {text_column!r} y {label_column!r}. "
            f"Columnas disponibles: {list(dataframe.columns)}"
        )

    output_dir = Path(output_dir)
    false_dir = output_dir / "Falso"
    true_dir = output_dir / "Verdad"
    false_dir.mkdir(parents=True, exist_ok=True)
    true_dir.mkdir(parents=True, exist_ok=True)

    counters = {0: 0, 1: 0}
    rows: list[dict[str, object]] = []
    for _, record in dataframe.iterrows():
        text = str(record[text_column]).strip()
        if not text or text.lower() == "nan":
            continue
        label = normalize_label(record[label_column])
        counters[label] += 1
        doc_id = build_doc_id(label, counters[label])
        destination = (false_dir if label == 0 else true_dir) / f"{doc_id}.txt"
        destination.write_text(text, encoding="utf-8")
        rows.append(
            {
                "doc_id": doc_id,
                "label": label,
                "source_type": "dataset",
                "source_name": source_name,
                "url": "",
                "title": text.splitlines()[0][:300],
                "date": "",
                "n_words": len(text.split()),
                "topic": "",
            }
        )

    metadata = pd.DataFrame(
        rows,
        columns=[
            "doc_id", "label", "source_type", "source_name", "url",
            "title", "date", "n_words", "topic",
        ],
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8")
    return metadata


def main(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    text_column: str = "text",
    label_column: str = "label",
    source_name: str = "dataset",
) -> pd.DataFrame:
    """Lee un CSV, genera corpusDataset y devuelve sus metadatos."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {input_path}")
    dataframe = pd.read_csv(input_path)
    metadata = write_dataset_corpus(
        dataframe, output_dir, text_column, label_column, source_name
    )
    print(f"Corpus de dataset generado en {output_dir}: {len(metadata)} documentos.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV de entrada.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--source-name", default="dataset")
    args = parser.parse_args()
    main(
        args.input,
        args.output_dir,
        args.text_column,
        args.label_column,
        args.source_name,
    )
