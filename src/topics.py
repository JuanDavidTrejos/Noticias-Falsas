from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

def get_lsa_topics(X, vectorizer, n_topics=5, n_words=10):
    """
    Aplica LSA (Latent Semantic Analysis) usando TruncatedSVD.
    Ideal para matrices continuas como TF-IDF.
    """
    lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa_matrix = lsa_model.fit_transform(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for _, topic in enumerate(lsa_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append(top_words)
        
    return lsa_model, lsa_matrix, topics

def get_lda_topics(X, vectorizer, n_topics=5, n_words=10):
    """
    Aplica LDA (Latent Dirichlet Allocation).
    Requiere matrices de frecuencias absolutas (BoW / TO).
    """
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_matrix = lda_model.fit_transform(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for _, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append(top_words)
        
    return lda_model, lda_matrix, topics