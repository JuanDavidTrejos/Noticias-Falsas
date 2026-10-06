import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Descargar stopwords si no están disponibles
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def get_spanish_stopwords():
    """Devuelve la lista de stopwords en español de NLTK."""
    return set(stopwords.words('spanish'))

def process_text_tokens(text, stop_words, stemmer, apply_stopwords=True, apply_stemming=True):
    """
    Aplica eliminación de stopwords y/o stemming a un texto individual según las banderas.
    """
    if not isinstance(text, str):
        return ""
    
    tokens = text.split()
    processed_tokens = []
    
    for word in tokens:
        # Filtrar stopword si la opción está activa
        if apply_stopwords and word.lower() in stop_words:
            continue
        
        # Aplicar stemming si la opción está activa
        if apply_stemming:
            word = stemmer.stem(word)
            
        processed_tokens.append(word)
        
    return " ".join(processed_tokens)

def preprocess_for_features(df, text_column, output_column='text_processed', apply_stopwords=True, apply_stemming=True):
    """
    Aplica la transformación a toda la columna del DataFrame, permitiendo activar/desactivar pasos.
    """
    stop_words = get_spanish_stopwords() if apply_stopwords else set()
    stemmer = SnowballStemmer('spanish') if apply_stemming else None
    
    # Aplicar la transformación creando la nueva columna
    df[output_column] = df[text_column].apply(
        lambda x: process_text_tokens(x, stop_words, stemmer, apply_stopwords, apply_stemming)
    )
    return df

def extract_features(df, text_column='text_processed', method='bow', min_freq=3):
    """
    Crea la matriz de características (BoW o TF-IDF) usando unigramas y bigramas,
    y eliminando términos con frecuencia menor a min_freq.
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_freq)
    else:
        # BoW clásico (frecuencias absolutas) requerido para el Punto 5
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=min_freq)
        
    matrix = vectorizer.fit_transform(df[text_column])
    
    return matrix, vectorizer