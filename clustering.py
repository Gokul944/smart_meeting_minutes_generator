from sklearn.cluster import AgglomerativeClustering

def cluster_speakers(embeddings, n_speakers):
    clustering = AgglomerativeClustering(n_clusters=n_speakers)
    labels = clustering.fit_predict(embeddings)
    return labels
