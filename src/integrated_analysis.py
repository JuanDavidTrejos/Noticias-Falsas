"""Resúmenes compactos para el análisis integrado del corpus."""

from __future__ import annotations

import pandas as pd


def compare_pos(pos_table: pd.DataFrame) -> pd.DataFrame:
    """Calcula la diferencia de proporciones POS: falsas menos verdaderas."""
    result = pos_table[["Verdaderas (%)", "Falsas (%)"]].copy()
    result["diferencia_falsas_menos_verdaderas"] = (
        result["Falsas (%)"] - result["Verdaderas (%)"]
    ).round(2)
    return result.sort_values(
        "diferencia_falsas_menos_verdaderas", ascending=False
    )


def compare_ner(ner_distribution: pd.DataFrame) -> pd.DataFrame:
    """Calcula la diferencia porcentual de tipos NER entre clases."""
    table = ner_distribution.pivot(
        index="tipo", columns="clase", values="porcentaje"
    ).fillna(0)
    table["diferencia_falsas_menos_verdaderas"] = (
        table.get("Falsas", 0) - table.get("Verdaderas", 0)
    ).round(2)
    return table.sort_values(
        "diferencia_falsas_menos_verdaderas", ascending=False
    )


def topics_table(topics, method: str) -> pd.DataFrame:
    """Convierte un diccionario o listas de palabras en tabla."""
    topic_items = topics.items() if isinstance(topics, dict) else enumerate(topics, 1)
    return pd.DataFrame(
        {
            "metodo": method,
            "topico": [
                key if isinstance(topics, dict) else f"Tópico {key}"
                for key, _ in topic_items
            ],
            "palabras": [
                ", ".join(words)
                for _, words in (
                    topics.items()
                    if isinstance(topics, dict)
                    else enumerate(topics, 1)
                )
            ],
        }
    )


def recommendations() -> pd.DataFrame:
    """Propone variables lingüísticas para una futura clasificación."""
    return pd.DataFrame(
        [
            {
                "señal": "Distribución POS",
                "variables": "proporción de NOUN, ADV y PRON",
                "uso": "añadir como características numéricas",
            },
            {
                "señal": "Entidades NER",
                "variables": "conteos por LOC, PER, ORG y MISC",
                "uso": "añadir frecuencias normalizadas por documento",
            },
            {
                "señal": "Temas",
                "variables": "distribución de tópicos LSA/LDA",
                "uso": "combinar con TF-IDF para capturar contexto semántico",
            },
        ]
    )
