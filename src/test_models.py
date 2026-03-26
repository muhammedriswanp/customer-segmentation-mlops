import joblib

# Test 1 — KMeans model
model = joblib.load('models/best_model.pkl')
print('KMeans loaded!')
print(f'Clusters: {model.n_clusters}')

# Test 2 — Best KMeans model
best_model = joblib.load('models/best_kmeans_model.pkl')
print('Best KMeans loaded!')
print(f'Best Clusters: {best_model.n_clusters}')
