"""Descarga, prepara y exporta el dataset de noticias falsas de Kaggle.

El resultado se guarda separadamente del corpus obtenido por scraping:

    corpus/
    ├── Falso/
    ├── Verdad/
    └── metadata.csv

Uso desde Python:
    from src.get_corpus_dataset import main
    metadata = main()

Uso desde PowerShell:
    py src/get_corpus_dataset.py --total 5000

Para usar un CSV local en lugar de Kaggle:
    py src/get_corpus_dataset.py --input data/noticias.csv
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_SLUG = "javieroterovizoso/spanish-political-fake-news"
DEFAULT_OUTPUT_DIR = ROOT / "corpus"
# 5.000 del dataset + 300 documentos de scraping = 5.300 en total.
DEFAULT_TOTAL = 5000
DEFAULT_SEED = 42
DEFAULT_MIN_WORDS = 20

# La práctica usa 1 para verdadera y 0 para falsa.
LABEL_FOLDERS = {1: "Verdad", 0: "Falso"}

# Los nombres se comparan después de quitar mayúsculas y tildes.
COLUMN_ALIASES = {
    "label": {"label", "etiqueta", "clase"},
    "title": {"titulo", "title", "headline"},
    "text": {"descripcion", "description", "texto", "text", "cuerpo"},
    "id": {"id"},
}


def normalize_column_name(name: object) -> str:
    """Normaliza un nombre de columna para compararlo con los alias."""
    normalized = unicodedata.normalize("NFKD", str(name))
    return "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    ).strip().lower()


def read_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV probando codificaciones y detectando automáticamente el separador."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer el archivo CSV: {path}")


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame | None:
    """Renombra las columnas relevantes a label/title/text/id."""
    mapping: dict[str, str] = {}
    for column in dataframe.columns:
        normalized = normalize_column_name(column)
        for destination, aliases in COLUMN_ALIASES.items():
            if normalized in aliases and destination not in mapping.values():
                mapping[column] = destination
                break

    required = {"label", "title", "text"}
    if not required.issubset(mapping.values()):
        return None

    standardized = dataframe.rename(columns=mapping)
    selected = ["label", "title", "text"]
    if "id" in standardized.columns:
        selected.append("id")
    return standardized[selected].copy()


def find_input_csv(
    input_path: Path | None,
    kaggle_slug: str,
) -> tuple[Path, pd.DataFrame]:
    """Obtiene el CSV local indicado o descarga y busca uno en Kaggle."""
    if input_path is not None:
        candidates = [Path(input_path)]
    else:
        try:
            import kagglehub
        except ImportError as error:
            raise ImportError(
                "Falta kagglehub. Instálalo con: py -m pip install kagglehub"
            ) from error

        downloaded_dir = Path(kagglehub.dataset_download(kaggle_slug))
        candidates = sorted(
            downloaded_dir.rglob("*.csv"),
            key=lambda path: path.stat().st_size,
            reverse=True,
        )

    for candidate in candidates:
        dataframe = standardize_columns(read_csv(candidate))
        if dataframe is not None:
            print(f"Archivo seleccionado: {candidate.name} | filas: {len(dataframe):,}")
            return candidate, dataframe

    raise ValueError(
        "Ningún CSV contiene las columnas esperadas: "
        "Label/etiqueta, Titulo/título y Descripcion/descripcion."
    )


def clean_dataset(
    dataframe: pd.DataFrame,
    min_words: int = DEFAULT_MIN_WORDS,
) -> pd.DataFrame:
    """Aplica limpieza estructural sin hacer aún el preprocesamiento de PLN."""
    original_count = len(dataframe)
    dataframe = dataframe.copy()

    # Solo se conservan las dos etiquetas válidas de la práctica.
    dataframe["label"] = pd.to_numeric(dataframe["label"], errors="coerce")
    dataframe = dataframe[dataframe["label"].isin([0, 1])].copy()
    dataframe["label"] = dataframe["label"].astype(int)

    # El título y la descripción se guardan unidos en el documento original.
    dataframe["title"] = dataframe["title"].fillna("").astype(str).str.strip()
    dataframe["text"] = dataframe["text"].fillna("").astype(str).str.strip()
    dataframe["text"] = (
        dataframe["title"].str.rstrip(". ")
        + ". "
        + dataframe["text"]
    ).str.strip(". ").str.strip()

    # Se eliminan registros sin suficiente contenido para el corpus.
    dataframe = dataframe[
        dataframe["text"].str.split().str.len() >= min_words
    ].copy()

    # Un mismo texto con dos etiquetas es ambiguo y se descarta completo.
    normalized_text = dataframe["text"].str.lower()
    conflicting = dataframe.groupby(normalized_text)["label"].transform("nunique") > 1
    dataframe = dataframe[~conflicting].copy()
    dataframe = dataframe[
        ~normalized_text.loc[dataframe.index].duplicated()
    ].copy()

    print(
        f"Limpieza estructural: {original_count:,} -> "
        f"{len(dataframe):,} noticias válidas"
    )
    return dataframe.reset_index(drop=True)


def balanced_sample(
    dataframe: pd.DataFrame,
    total: int = DEFAULT_TOTAL,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Selecciona la misma cantidad de noticias verdaderas y falsas."""
    if total <= 0 or total % 2:
        raise ValueError("total debe ser un número positivo y par.")

    per_class = total // 2
    available = dataframe["label"].value_counts()
    print(
        "Disponibles por clase:",
        {
            LABEL_FOLDERS[label]: int(available.get(label, 0))
            for label in LABEL_FOLDERS
        },
    )
    if any(available.get(label, 0) < per_class for label in LABEL_FOLDERS):
        raise ValueError(
            f"No hay {per_class} noticias disponibles en cada clase; "
            "reduce el total solicitado."
        )

    parts = [
        group.sample(n=per_class, random_state=seed)
        for _, group in dataframe.groupby("label")
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def export_corpus(
    dataframe: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_name: str = "fuente2_kaggle",
    clean_previous: bool = True,
) -> pd.DataFrame:
    """Escribe un TXT por noticia y un metadata.csv con formato común."""
    output_dir = Path(output_dir)
    folders = {label: output_dir / folder for label, folder in LABEL_FOLDERS.items()}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        if clean_previous:
            for old_file in folder.glob("ds_*.txt"):
                old_file.unlink()

    rows: list[dict[str, object]] = []
    counters = {0: 0, 1: 0}
    for record in dataframe.itertuples(index=False):
        label = int(record.label)
        counters[label] += 1
        folder_name = LABEL_FOLDERS[label]
        prefix = "ds_falso" if label == 0 else "ds_verdad"
        filename = f"{prefix}_{counters[label]:04d}.txt"
        text = str(record.text).strip()
        (folders[label] / filename).write_text(text, encoding="utf-8")

        original_id = getattr(record, "id", "")
        rows.append(
            {
                "doc_id": filename[:-4],
                "label": label,
                "source_type": "dataset",
                "source_name": source_name,
                "url": "",
                "title": str(record.title)[:300],
                "date": "",
                "n_words": len(text.split()),
                "topic": "politica",
                "file": f"{folder_name}/{filename}",
                "original_id": original_id,
            }
        )

    metadata = pd.DataFrame(rows)
    metadata_path = output_dir / "metadata.csv"
    # Conserva los documentos de scraping ya existentes. Solo se reemplazan
    # las filas y archivos pertenecientes al dataset (prefijo ds_).
    if metadata_path.exists():
        previous = pd.read_csv(metadata_path)
        previous = previous[
            ~previous["doc_id"].astype(str).str.startswith("ds_")
        ]
        metadata = pd.concat([previous, metadata], ignore_index=True, sort=False)
        metadata = metadata.sort_values("doc_id").reset_index(drop=True)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")
    return metadata


def main(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    total: int = DEFAULT_TOTAL,
    seed: int = DEFAULT_SEED,
    min_words: int = DEFAULT_MIN_WORDS,
    kaggle_slug: str = DEFAULT_KAGGLE_SLUG,
) -> pd.DataFrame:
    """Ejecuta descarga/lectura, limpieza, muestreo balanceado y exportación."""
    _, raw_dataframe = find_input_csv(input_path, kaggle_slug)
    cleaned_dataframe = clean_dataset(raw_dataframe, min_words)
    sample = balanced_sample(cleaned_dataframe, total, seed)
    metadata = export_corpus(sample, output_dir)

    print("\nArchivos generados por carpeta:")
    print(metadata["label"].map(LABEL_FOLDERS).value_counts().to_string())
    print(f"Total: {len(metadata):,} TXT en '{Path(output_dir)}'")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV local; si se omite, usa Kaggle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--kaggle-slug", default=DEFAULT_KAGGLE_SLUG)
    return parser


if __name__ == "__main__":
    arguments = _build_parser().parse_args()
    try:
        main(
            input_path=arguments.input,
            output_dir=arguments.output_dir,
            total=arguments.total,
            seed=arguments.seed,
            min_words=arguments.min_words,
            kaggle_slug=arguments.kaggle_slug,
        )
    except (FileNotFoundError, ImportError, ValueError) as error:
        sys.exit(f"Error: {error}")
