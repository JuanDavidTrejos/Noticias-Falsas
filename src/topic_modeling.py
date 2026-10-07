import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from src.features import get_spanish_stopwords

# Palabras de plantilla de las fuentes scrapeadas
SOURCE_BOILERPLATE = {"bbc", "mundo", "newtral", "publicidad", "getty", "afp",
                      "fuente", "imagen", "imagenes", "lectura", "recomendada"}

# Mismos parámetros en la coherencia y en el modelo final
LDA_PARAMS = dict(random_state=42, passes=10, alpha='auto')


def run_lsa(texts, num_topics, top_n=10, extra_stopwords=None):
    stopwords_es = list(get_spanish_stopwords(extra_stopwords))
    vectorizer = TfidfVectorizer(stop_words=stopwords_es, max_df=0.85, min_df=5)
    X = vectorizer.fit_transform(texts)

    lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_model.fit(X)

    terms = vectorizer.get_feature_names_out()
    topics = {}
    for i, comp in enumerate(lsa_model.components_):
        idx = comp.argsort()[::-1][:top_n]
        topics[f"Tema {i+1}"] = [terms[j] for j in idx]
    return lsa_model, vectorizer, topics


def prepare_gensim_corpus(texts, extra_stopwords=None, min_len=2):
    stopwords_es = get_spanish_stopwords(extra_stopwords)
    tokenized_texts = [
        [w for w in str(t).split() if w not in stopwords_es and len(w) >= min_len]
        for t in texts
    ]
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in tokenized_texts]
    return dictionary, corpus, tokenized_texts


def run_lda(corpus, dictionary, num_topics, top_n=10):
    lda_model = LdaModel(corpus=corpus, id2word=dictionary,
                         num_topics=num_topics, **LDA_PARAMS)
    topics = {}
    for idx, topic in lda_model.show_topics(formatted=False, num_topics=num_topics, num_words=top_n):
        topics[f"Tema {idx+1}"] = [w[0] for w in topic]
    return lda_model, topics


def calculate_coherence_values(dictionary, corpus, texts, start=2, limit=10, step=1):
    coherence_values = []
    for k in range(start, limit + 1, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, **LDA_PARAMS)
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                            coherence='c_v', processes=1)
        coherence_values.append(cm.get_coherence())
    return coherence_values


def plot_coherence(start, limit, step, coherence_values):
    x = range(start, limit + 1, step)
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Número de temas (k)")
    plt.ylabel("Score de Coherencia (c_v)")
    plt.title("Número de temas (k) vs. Score de Coherencia")
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()