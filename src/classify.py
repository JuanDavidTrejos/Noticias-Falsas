from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import get_spanish_stopwords


REDUCTIONS = {
    "Ninguna": (False, False),
    "Solo stopwords": (True, False),
    "Solo stemming": (False, True),
    "Stopwords + stemming": (True, True),
}

ALGORITHMS = {
    "Regresión logística": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Árbol de decisión": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="linear", random_state=42),
}


def _normalize_texts(texts, use_stopwords, use_stemming):
    stopwords = get_spanish_stopwords() if use_stopwords else set()
    stemmer = None
    if use_stemming:
        from nltk.stem import SnowballStemmer

        stemmer = SnowballStemmer("spanish")

    normalized = []
    for text in texts:
        words = str(text).split()
        if use_stopwords:
            words = [word for word in words if word not in stopwords]
        if stemmer is not None:
            words = [stemmer.stem(word) for word in words]
        normalized.append(" ".join(words))
    return normalized


def _vectorizer(weighting, min_df):
    cls = CountVectorizer if weighting == "TO" else TfidfVectorizer
    return cls(ngram_range=(1, 2), min_df=min_df)


def evaluate_classifiers(
    data,
    text_column="clean_text",
    label_column="label",
    test_size=0.2,
    cv=10,
    min_df=3,
    random_state=42,
):
    """Evalúa 4 algoritmos x 4 reducciones x 2 ponderaciones.

    El vectorizador se ajusta únicamente con entrenamiento. La CV de 10
    pliegues se ejecuta sobre ese 80%; el 20% reservado se evalúa aparte.
    """
    texts = data[text_column].fillna("").astype(str).tolist()
    labels = data[label_column].astype(int).to_numpy()
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    splitter = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    rows = []

    for reduction, (use_stopwords, use_stemming) in REDUCTIONS.items():
        train_reduced = _normalize_texts(
            train_texts, use_stopwords, use_stemming
        )
        test_reduced = _normalize_texts(test_texts, use_stopwords, use_stemming)
        for weighting in ("TO", "TF-IDF"):
            vectorizer = _vectorizer(weighting, min_df)
            x_train = vectorizer.fit_transform(train_reduced)
            x_test = vectorizer.transform(test_reduced)
            for algorithm, estimator in ALGORITHMS.items():
                scores = cross_validate(
                    estimator,
                    x_train,
                    y_train,
                    cv=splitter,
                    scoring={
                        "accuracy": "accuracy",
                        "precision": "precision_weighted",
                        "recall": "recall_weighted",
                        "f1": "f1_weighted",
                    },
                    n_jobs=1,
                )
                fitted = estimator.fit(x_train, y_train)
                predictions = fitted.predict(x_test)
                from sklearn.metrics import (
                    accuracy_score,
                    f1_score,
                    precision_score,
                    recall_score,
                )

                rows.append(
                    {
                        "algoritmo": algorithm,
                        "reduccion": reduction,
                        "ponderacion": weighting,
                        "n_caracteristicas": x_train.shape[1],
                        "accuracy_cv": scores["test_accuracy"].mean(),
                        "precision_cv": scores["test_precision"].mean(),
                        "recall_cv": scores["test_recall"].mean(),
                        "f1_cv": scores["test_f1"].mean(),
                        "accuracy_test": accuracy_score(y_test, predictions),
                        "precision_test": precision_score(
                            y_test, predictions, average="weighted", zero_division=0
                        ),
                        "recall_test": recall_score(
                            y_test, predictions, average="weighted", zero_division=0
                        ),
                        "f1_test": f1_score(
                            y_test, predictions, average="weighted", zero_division=0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_max_f1_by_algorithm(results):
    summary = results.groupby("algoritmo", as_index=False)["f1_test"].max()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(summary["algoritmo"], summary["f1_test"], color="#2878b5")
    axis.set_title("F1-score máximo por algoritmo")
    axis.set_ylabel("F1-score ponderado (test)")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure


def plot_mean_f1_by_weighting(results):
    summary = (
        results.groupby(["ponderacion", "algoritmo"], as_index=False)["f1_test"]
        .mean()
    )
    return _grouped_bar(summary, "ponderacion", "F1-score medio por ponderación y algoritmo")


def plot_mean_f1_by_reduction(results):
    summary = (
        results.groupby(["reduccion", "algoritmo"], as_index=False)["f1_test"]
        .mean()
    )
    return _grouped_bar(summary, "reduccion", "F1-score medio por reducción y algoritmo")


def _grouped_bar(summary, category, title):
    categories = list(summary[category].unique())
    algorithms = list(summary["algoritmo"].unique())
    positions = range(len(categories))
    width = 0.8 / len(algorithms)
    figure, axis = plt.subplots(figsize=(11, 6))
    for index, algorithm in enumerate(algorithms):
        values = [
            summary.loc[
                (summary[category] == item)
                & (summary["algoritmo"] == algorithm),
                "f1_test",
            ].iloc[0]
            for item in categories
        ]
        axis.bar(
            [p + index * width for p in positions],
            values,
            width,
            label=algorithm,
        )
    axis.set_xticks(
        [p + width * (len(algorithms) - 1) / 2 for p in positions],
        categories,
        rotation=20,
        ha="right",
    )
    axis.set_title(title)
    axis.set_ylabel("F1-score ponderado (test)")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    return figure


def save_classification_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "classification_results.csv", index=False)
    figures = {
        "f1_max_algorithm.png": plot_max_f1_by_algorithm(results),
        "f1_mean_weighting_algorithm.png": plot_mean_f1_by_weighting(results),
        "f1_mean_reduction_algorithm.png": plot_mean_f1_by_reduction(results),
    }
    paths = {"table": output_dir / "classification_results.csv"}
    for filename, figure in figures.items():
        path = output_dir / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths[filename] = path
    return paths
