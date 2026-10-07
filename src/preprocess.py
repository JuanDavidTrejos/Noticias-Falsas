"""Preprocesamiento lingüístico del corpus para la fase 2 de la práctica.

Este módulo no modifica los archivos TXT originales. Sus funciones reciben
texto y devuelven una nueva cadena para usarla en el notebook o en modelos.
"""

from __future__ import annotations

import re
import unicodedata
import warnings

from bs4 import BeautifulSoup

try:  # bs4 >= 4.11
    from bs4 import MarkupResemblesLocatorWarning

    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
except ImportError:
    pass


# Dominios (la terminación se compara en minúsculas para no borrar "sí.Es ...").
_TLDS = r"(?-i:com|co|org|net|edu|es|gob|gov|io|info|tv)"

# URLs (http/www), correos electrónicos y dominios sueltos como newtral.es
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    rf"|(?<![@\w])[\w-]+(?:\.[\w-]+)*\.{_TLDS}\b(?:/\S*)?",
    re.IGNORECASE,
)

# Siglas con puntos: EE.UU. -> EEUU, S.A. -> SA (se aplica antes de pasar a minúsculas)
ACRONYM_PATTERN = re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ]{1,3}\.){2,}")

# Números enteros/decimales, incluso pegados a letras (covid19), con º ª % opcionales
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*[ºª°%]?")

# Emoticones aislados (no dentro de palabras): :) ;D =( :-P :'( xD <3
EMOTICON_PATTERN = re.compile(
    r"(?<!\S)(?:[:;=][-'^o]?[)(DPp/\\O*]+|[xX][dD]|<3)(?!\w)"
)

# Puntuación y símbolos (incluye emojis y guion bajo)
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]|_", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")

# Marcadores temporales para proteger la ñ al quitar tildes
_N_LOWER, _N_UPPER = "\ue000", "\ue001"


def to_lowercase(text: str) -> str:
    """Convierte el texto a minúsculas."""
    return str(text).lower()


def remove_accents(text: str) -> str:
    """Elimina tildes de las vocales (á->a, ü->u) y CONSERVA la ñ."""
    text = str(text).replace("ñ", _N_LOWER).replace("Ñ", _N_UPPER)
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    stripped = unicodedata.normalize("NFC", stripped)
    return stripped.replace(_N_LOWER, "ñ").replace(_N_UPPER, "Ñ")


def remove_urls(text: str) -> str:
    """Elimina URLs, correos y dominios web."""
    return URL_PATTERN.sub(" ", str(text))


def join_acronyms(text: str) -> str:
    """Quita los puntos de las siglas: EE.UU. -> EEUU."""
    return ACRONYM_PATTERN.sub(lambda m: m.group(0).replace(".", ""), str(text))


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
    """Elimina signos de puntuación y símbolos; conserva letras, números y espacios."""
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

    Orden: HTML -> enlaces -> siglas -> emoticones -> números -> minúsculas
    -> tildes (conserva ñ) -> puntuación -> saltos de línea y espacios.
    """
    result = str(text)
    if remove_html_:
        result = remove_html(result)
    if remove_urls_:
        result = remove_urls(result)
    if remove_punctuation_:
        result = join_acronyms(result)  # antes de borrar los puntos
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