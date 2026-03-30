import joblib

# Test 1 — Best model (Agglomerative)
model1 = joblib.load('models/best_model.pkl')
print(f'best_model.pkl      → {type(model1).__name__}, k={model1.n_clusters}')

# Test 2 — KMeans model
model2 = joblib.load('models/kmeans_model.pkl')
print(f'kmeans_model.pkl    → {type(model2).__name__}, k={model2.n_clusters}')

# Test 3 — Scaler
scaler = joblib.load('models/scaler.pkl')
print(f'scaler.pkl          → {type(scaler).__name__}')

# Test 4 — PCA
pca = joblib.load('models/pca.pkl')
print(f'pca.pkl             → {type(pca).__name__}, components={pca.n_components_}')

print("\nAll models loaded successfully!")
