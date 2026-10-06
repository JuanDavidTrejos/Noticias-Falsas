"""Preprocesamiento lingüístico del corpus para la fase 2 de la práctica.

Este módulo no modifica los archivos TXT originales. Sus funciones reciben
texto y devuelven una nueva cadena para usarla en el notebook o en modelos.
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup


# URLs completas, dominios con www y enlaces que comienzan por http(s).
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|(?<![@\w])[\w.-]+\.(?:com|co|org|net|edu)(?:/\S*)?",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)*\b")
EMOTICON_PATTERN = re.compile(
    r"(?::|;|=|8)(?:-|'|^)?(?:\)|\(|D|P|p|/|\\|O|\*)+"
)
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def to_lowercase(text: str) -> str:
    """Convierte el texto a minúsculas."""
    return str(text).lower()


def remove_accents(text: str) -> str:
    """Elimina tildes conservando la letra base, por ejemplo á -> a."""
    normalized = unicodedata.normalize("NFD", str(text))
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def remove_urls(text: str) -> str:
    """Elimina URLs, dominios web y rutas asociadas."""
    return URL_PATTERN.sub(" ", str(text))


def remove_numbers(text: str) -> str:
    """Elimina cifras enteras y decimales."""
    return NUMBER_PATTERN.sub(" ", str(text))


def remove_html(text: str) -> str:
    """Convierte HTML en texto y elimina sus etiquetas."""
    return BeautifulSoup(str(text), "html.parser").get_text(" ")


def remove_line_breaks(text: str) -> str:
    """Convierte saltos de línea y retornos de carro en espacios."""
    return re.sub(r"[\r\n\t]+", " ", str(text))


def remove_emoticons(text: str) -> str:
    """Elimina emoticones ASCII frecuentes."""
    return EMOTICON_PATTERN.sub(" ", str(text))


def remove_punctuation(text: str) -> str:
    """Elimina signos de puntuación y conserva letras, números y espacios."""
    return PUNCTUATION_PATTERN.sub(" ", str(text))


def normalize_spaces(text: str) -> str:
    """Reduce espacios consecutivos y elimina espacios en los extremos."""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_accents_: bool = True,
    remove_numbers_: bool = True,
    remove_urls_: bool = True,
    remove_html_: bool = True,
    remove_punctuation_: bool = True,
    remove_emoticons_: bool = True,
) -> str:
    """Aplica el pipeline completo de limpieza solicitado en la rúbrica.

    Los parámetros permiten activar o desactivar pasos para comparar
    variantes en el notebook sin duplicar código.
    """
    result = str(text)
    if remove_html_:
        result = remove_html(result)
    if remove_urls_:
        result = remove_urls(result)
    if remove_emoticons_:
        result = remove_emoticons(result)
    if remove_numbers_:
        result = remove_numbers(result)
    if lowercase:
        result = to_lowercase(result)
    if remove_accents_:
        result = remove_accents(result)
    if remove_punctuation_:
        result = remove_punctuation(result)
    return normalize_spaces(remove_line_breaks(result))


def preprocess_dataframe(
    dataframe,
    text_column: str = "text",
    output_column: str = "clean_text",
):
    """Devuelve una copia del DataFrame con el texto preprocesado."""
    if text_column not in dataframe.columns:
        raise KeyError(f"No existe la columna de texto: {text_column}")
    result = dataframe.copy()
    result[output_column] = result[text_column].fillna("").map(clean_text)
    return result
