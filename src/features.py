import unicodedata
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.preprocess import remove_accents as strip_accents_keep_n

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def get_spanish_stopwords(extra=None):
    """Stopwords de NLTK en versión original Y sin tildes (el corpus ya no tiene tildes)."""
    base = set(stopwords.words('spanish'))
    sw = base | {strip_accents_keep_n(w) for w in base}
    if extra:
        sw |= {strip_accents_keep_n(w.lower()) for w in extra}
    return sw


def process_text_tokens(text, stop_words, stemmer, apply_stopwords=True, apply_stemming=True):
    if not isinstance(text, str):
        return ""
    out = []
    for word in text.split():
        if apply_stopwords and word.lower() in stop_words:
            continue
        if apply_stemming:
            word = stemmer.stem(word)
        out.append(word)
    return " ".join(out)


def preprocess_for_features(df, text_column, output_column='text_processed',
                            apply_stopwords=True, apply_stemming=True, extra_stopwords=None):
    df = df.copy()  # no mutar el DataFrame original
    stop_words = get_spanish_stopwords(extra_stopwords) if apply_stopwords else set()
    stemmer = SnowballStemmer('spanish') if apply_stemming else None
    df[output_column] = df[text_column].apply(
        lambda x: process_text_tokens(x, stop_words, stemmer, apply_stopwords, apply_stemming)
    )
    return df


def build_vectorizer(texts, method='bow', min_freq=3):
    """Vocabulario = unigramas+bigramas con frecuencia TOTAL >= min_freq.
    Mismo vocabulario para BoW y TF-IDF."""
    cv = CountVectorizer(ngram_range=(1, 2))
    counts = cv.fit_transform(texts)
    freq = np.asarray(counts.sum(axis=0)).ravel()
    vocab = cv.get_feature_names_out()[freq >= min_freq]
    cls = TfidfVectorizer if method == 'tfidf' else CountVectorizer
    vectorizer = cls(ngram_range=(1, 2), vocabulary=vocab)
    vectorizer.fit(texts)
    return vectorizer


def extract_features(df, text_column='text_processed', method='bow', min_freq=3):
    vectorizer = build_vectorizer(df[text_column], method, min_freq)
    return vectorizer.transform(df[text_column]), vectorizer