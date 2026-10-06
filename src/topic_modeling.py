import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from spacy.lang.es.stop_words import STOP_WORDS

def get_spanish_stopwords():
    """Devuelve la lista de stop words en español de spaCy."""
    return list(STOP_WORDS)

# ==========================================
# 7.1 LSA (Latent Semantic Analysis)
# ==========================================
def run_lsa(texts, num_topics, top_n=10):
    """
    Ejecuta LSA sobre una lista de textos.
    Aplica TF-IDF y luego TruncatedSVD.
    """
    stopwords_es = get_spanish_stopwords()
    
    # Representación de documentos con TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords_es, max_df=0.85, min_df=5)
    X = vectorizer.fit_transform(texts)
    
    # Reducción de dimensionalidad
    lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_model.fit(X)
    
    terms = vectorizer.get_feature_names_out()
    
    # Extraer las palabras principales por tema
    topics = {}
    for i, comp in enumerate(lsa_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_n]
        topics[f"Tema {i+1}"] = [t[0] for t in sorted_terms]
        
    return lsa_model, vectorizer, topics

# ==========================================
# Preparación para LDA y Coherencia (Gensim)
# ==========================================
def prepare_gensim_corpus(texts):
    """
    Tokeniza, elimina stopwords y crea el Diccionario y el Corpus Bag of Words para Gensim.
    """
    stopwords_es = set(get_spanish_stopwords())
    
    # Tokenización simple y filtrado de stopwords
    tokenized_texts = [
        [word for word in str(text).split() if word not in stopwords_es and len(word) > 2] 
        for text in texts
    ]
    
    # Creación del diccionario
    dictionary = corpora.Dictionary(tokenized_texts)
    # Filtrar palabras que aparecen en menos de 5 documentos o en más del 50%
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # Representación Bag of Words
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    return dictionary, corpus, tokenized_texts

# ==========================================
# 7.2 LDA (Latent Dirichlet Allocation)
# ==========================================
def run_lda(corpus, dictionary, num_topics, top_n=10):
    """
    Ejecuta el modelo LDA usando Gensim.
    """
    lda_model = LdaModel(
        corpus=corpus, 
        id2word=dictionary, 
        num_topics=num_topics, 
        random_state=42, 
        passes=10,
        alpha='auto'
    )
    
    topics = {}
    for idx, topic in lda_model.show_topics(formatted=False, num_topics=num_topics, num_words=top_n):
        topics[f"Tema {idx+1}"] = [w[0] for w in topic]
        
    return lda_model, topics

# ==========================================
# 7.3 Cálculo de Coherencia y Gráfica
# ==========================================
def calculate_coherence_values(dictionary, corpus, texts, start=2, limit=10, step=1):
    """
    Calcula el c_v coherence score para distintos valores de k.
    """
    coherence_values = []
    
    for num_topics in range(start, limit + 1, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=5)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
    return coherence_values

def plot_coherence(start, limit, step, coherence_values):
    """
    Genera la gráfica de k vs. Coherencia.
    """
    x = range(start, limit + 1, step)
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Número de temas (k)")
    plt.ylabel("Score de Coherencia (c_v)")
    plt.title("Número de temas (k) vs. Score de Coherencia")
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()