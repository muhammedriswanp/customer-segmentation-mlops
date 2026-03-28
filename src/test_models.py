import joblib

# Test 1 — KMeans model
model = joblib.load('models/best_model.pkl')
print('KMeans loaded!')
print(f'Clusters: {model.n_clusters}')

