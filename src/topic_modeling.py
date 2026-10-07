import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from gensim import corpora
    from gensim.models import CoherenceModel, LdaModel
except ImportError:
    corpora = None
    CoherenceModel = None
    LdaModel = None

from src import features

SOURCE_BOILERPLATE = {
    "bbc", "mundo", "newtral", "publicidad", "getty", "afp",
    "fuente", "imagen", "imagenes", "lectura", "recomendada",
}
LDA_PARAMS = dict(random_state=42, passes=10, alpha="auto")


def _stopwords(extra_stopwords=None):
    """Obtiene stopwords incluso si el kernel conserva una API anterior."""
    try:
        return features.get_spanish_stopwords(extra_stopwords)
    except TypeError:
        words = features.get_spanish_stopwords()
        if extra_stopwords:
            words.update(str(word).lower() for word in extra_stopwords)
        return words


def run_lsa(texts, num_topics, top_n=10, extra_stopwords=None):
    """Ejecuta LSA sobre TF-IDF y devuelve el modelo, vectorizador y tópicos."""
    stopwords = list(_stopwords(extra_stopwords))
    vectorizer = TfidfVectorizer(
        stop_words=stopwords, max_df=0.85, min_df=5
    )
    matrix = vectorizer.fit_transform(texts)
    model = TruncatedSVD(n_components=num_topics, random_state=42).fit(matrix)
    terms = vectorizer.get_feature_names_out()
    topics = {
        f"Tema {index + 1}": [
            terms[i] for i in component.argsort()[::-1][:top_n]
        ]
        for index, component in enumerate(model.components_)
    }
    return model, vectorizer, topics


def prepare_gensim_corpus(texts, extra_stopwords=None, min_len=2):
    """Prepara diccionario y corpus BoW para Gensim."""
    if corpora is None:
        raise ImportError(
            "Gensim no está disponible. Se usará prepare_sklearn_corpus."
        )
    stopwords = _stopwords(extra_stopwords)
    tokenized = [
        [word for word in str(text).split()
         if word not in stopwords and len(word) >= min_len]
        for text in texts
    ]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    return dictionary, corpus, tokenized


def prepare_sklearn_corpus(texts, extra_stopwords=None):
    """Prepara una matriz BoW para LDA cuando Gensim no está disponible."""
    stopwords = _stopwords(extra_stopwords)
    cleaned = [
        " ".join(
            word for word in str(text).split()
            if word not in stopwords and len(word) >= 2
        )
        for text in texts
    ]
    vectorizer = TfidfVectorizer(use_idf=False, norm=None, min_df=5)
    matrix = vectorizer.fit_transform(cleaned)
    return matrix, vectorizer, [text.split() for text in cleaned]


def run_lda(corpus, dictionary, num_topics, top_n=10):
    """Ejecuta LDA con Gensim o con scikit-learn como respaldo."""
    if LdaModel is None:
        model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method="batch",
        ).fit(corpus)
        terms = dictionary.get_feature_names_out()
        topics = {
            f"Tema {index + 1}": [
                terms[i] for i in component.argsort()[-top_n:][::-1]
            ]
            for index, component in enumerate(model.components_)
        }
        return model, topics

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        **LDA_PARAMS,
    )
    topics = {
        f"Tema {index + 1}": [word for word, _ in words]
        for index, words in model.show_topics(
            formatted=False, num_topics=num_topics, num_words=top_n
        )
    }
    return model, topics


def calculate_coherence_values(
    dictionary, corpus, texts, start=2, limit=10, step=1
):
    """Calcula coherencia Gensim o una puntuación comparable con sklearn."""
    if LdaModel is None:
        scores = []
        for topics in range(start, limit + 1, step):
            model = LatentDirichletAllocation(
                n_components=topics,
                random_state=42,
                learning_method="batch",
            ).fit(corpus)
            scores.append(float(model.score(corpus)))
        return scores

    scores = []
    for topics in range(start, limit + 1, step):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=topics,
            **LDA_PARAMS,
        )
        coherence = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )
        scores.append(coherence.get_coherence())
    return scores


def plot_coherence(start, limit, step, coherence_values):
    """Devuelve una figura de k frente a la puntuación de coherencia."""
    x_values = list(range(start, limit + 1, step))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        x_values,
        coherence_values,
        marker="o",
        linestyle="-",
        color="b",
    )
    axis.set_xlabel("Número de temas (k)")
    axis.set_ylabel("Score de Coherencia (c_v)")
    axis.set_title("Número de temas (k) vs. Score de Coherencia")
    axis.set_xticks(x_values)
    axis.grid(True, linestyle="--", alpha=0.7)
    figure.tight_layout()
    return figure
