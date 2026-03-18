import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score



def apply_pca(X, variance_threshold=0.9):
   
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def compute_elbow(X, k_range=range(1, 11)):
   
    wcss = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)# runs 10 times with different random starting points and picks the best result.
        model.fit(X)
        wcss.append(model.inertia_)

    return wcss


def compute_silhouette(X, model_class, k_range=range(3, 7),  **model_params):
    
    scores = {}

    for k in k_range:
        model = model_class(n_clusters=k, **model_params)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score

    return scores


def run_kmeans(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def plot_silhouette_scores(X, k_range=range(2, 10)):
    scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        scores.append(silhouette_score(X, labels))

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), scores, marker='s', color='green', linewidth=2)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by K')
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.3)
    plt.savefig('../outputs/reports/silhouette_scores.png', dpi=150, bbox_inches='tight')
    plt.show()
    return scores
    
def run_hierarchical(X, n_clusters=4, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return model, labels
