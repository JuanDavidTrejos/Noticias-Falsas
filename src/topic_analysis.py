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
    df, topic_column="dominant_topic", label_column="label",
    topic_words=None, interpretations=None
):
    """Tabla de distribución de temas. topic_words e interpretations son dicts {nombre_tema: ...}."""
    df_temp = df.copy()
    df_temp["_label_str"] = df_temp[label_column].map({1: "Verdaderas", 0: "Falsas"})

    crosstab = pd.crosstab(df_temp[topic_column], df_temp["_label_str"])
    for col in ["Verdaderas", "Falsas"]:
        if col not in crosstab.columns:
            crosstab[col] = 0

    total_v = crosstab["Verdaderas"].sum()
    total_f = crosstab["Falsas"].sum()

    table_df = pd.DataFrame({
        "Tema": crosstab.index,
        "Verdaderas (cantidad)": crosstab["Verdaderas"].values,
        "Verdaderas (%)": np.round(crosstab["Verdaderas"].values / total_v * 100, 2),
        "Falsas (cantidad)": crosstab["Falsas"].values,
        "Falsas (%)": np.round(crosstab["Falsas"].values / total_f * 100, 2),
    }).reset_index(drop=True)

    if topic_words:
        table_df.insert(1, "Palabras representativas",
                        table_df["Tema"].map(lambda t: ", ".join(topic_words.get(t, []))))
    if interpretations:
        table_df["Interpretación"] = table_df["Tema"].map(interpretations)
    return table_df


def plot_topic_comparison(topic_table, title="Distribución Porcentual de Temas según Veracidad"):
    df_plot = pd.melt(topic_table, id_vars=["Tema"],
                      value_vars=["Verdaderas (%)", "Falsas (%)"],
                      var_name="Clase", value_name="Porcentaje")
    df_plot["Clase"] = df_plot["Clase"].str.replace(" (%)", "", regex=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot, x="Tema", y="Porcentaje", hue="Clase",
                palette={"Verdaderas": "#27ae60", "Falsas": "#c0392b"},
                edgecolor="black", linewidth=0.8, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Tema Dominante", fontsize=11, fontweight="bold")
    ax.set_ylabel("Proporción en la clase (%)", fontsize=11, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Veracidad de Noticia", frameon=True)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, 3), textcoords="offset points")
    fig.tight_layout()
    return fig  