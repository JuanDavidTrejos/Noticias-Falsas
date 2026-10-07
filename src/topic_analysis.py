from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_dominant_topic(doc_topic_matrix, topic_labels=None):
    """Obtiene el tema dominante para cada documento a partir de la matriz documento-tema.

    Parameters:
    -----------
    doc_topic_matrix : numpy.ndarray o pd.DataFrame
        Matriz de probabilidades o pesos de temas por documento (p. ej. de
        LDA o NMF).
    topic_labels : list o dict, opcional
        Nombres o etiquetas asignadas a cada índice de tema.

    Returns:
    --------
    pd.Series
        Serie con el tema dominante para cada documento.
    """
    if isinstance(doc_topic_matrix, pd.DataFrame):
        matrix_vals = doc_topic_matrix.values
    else:
        matrix_vals = np.array(doc_topic_matrix)

    dominant_indices = np.argmax(matrix_vals, axis=1)

    if topic_labels is not None:
        if isinstance(topic_labels, dict):
            return pd.Series(
                [topic_labels.get(idx, f"Tema_{idx}") for idx in dominant_indices]
            )
        elif isinstance(topic_labels, list):
            return pd.Series([topic_labels[idx] for idx in dominant_indices])

    return pd.Series([f"Tema_{idx}" for idx in dominant_indices])


def compute_topic_distribution_table(
    df, topic_column="dominant_topic", label_column="label"
):
    """Genera la tabla de distribución de temas entre noticias Verdaderas y Falsas

    con la estructura exigida en el enunciado.
    """
    df_temp = df.copy()

    # Mapeo explicito de etiquetas si vienen numericas (1=Verdaderas, 0=Falsas)
    if df_temp[label_column].dtype in [int, float, np.int64]:
        label_map = {1: "Verdaderas", 0: "Falsas"}
        df_temp["_label_str"] = df_temp[label_column].map(label_map)
    else:
        df_temp["_label_str"] = df_temp[label_column]

    # Tabla cruzada de conteos
    crosstab = pd.crosstab(df_temp[topic_column], df_temp["_label_str"])

    for col in ["Verdaderas", "Falsas"]:
        if col not in crosstab.columns:
            crosstab[col] = 0

    total_verdaderas = crosstab["Verdaderas"].sum()
    total_falsas = crosstab["Falsas"].sum()

    # Construccion de la tabla final
    table_df = pd.DataFrame(
        {
            "Tema": crosstab.index,
            "Verdaderas (cantidad)": crosstab["Verdaderas"].values,
            "Verdaderas (%)": np.round(
                (crosstab["Verdaderas"].values / total_verdaderas) * 100, 2
            )
            if total_verdaderas > 0
            else 0.0,
            "Falsas (cantidad)": crosstab["Falsas"].values,
            "Falsas (%)": np.round(
                (crosstab["Falsas"].values / total_falsas) * 100, 2
            )
            if total_falsas > 0
            else 0.0,
        }
    )

    return table_df.reset_index(drop=True)


def plot_topic_comparison(
    topic_table, title="Distribución Porcentual de Temas según Veracidad"
):
    """Genera un gráfico de barras comparativo de la distribución porcentual de temas

    para Noticias Verdaderas vs Noticias Falsas.
    """
    df_plot = pd.melt(
        topic_table,
        id_vars=["Tema"],
        value_vars=["Verdaderas (%)", "Falsas (%)"],
        var_name="Clase",
        value_name="Porcentaje",
    )

    df_plot["Clase"] = df_plot["Clase"].str.replace(" (%)", "", regex=False)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    palette = {"Verdaderas": "#27ae60", "Falsas": "#c0392b"}

    ax = sns.barplot(
        data=df_plot,
        x="Tema",
        y="Porcentaje",
        hue="Clase",
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Tema Dominante", fontsize=11, fontweight="bold")
    plt.ylabel("Proporción en la clase (%)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Veracidad de Noticia", frameon=True)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.show()