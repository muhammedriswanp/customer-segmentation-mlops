import pandas as pd
import mlflow
import joblib
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from  preprocessing import preprocess
from feature_engineering import engineer_features, scale_features
from clustering import apply_pca

# Prepare data
print("Loading data...")
df = pd.read_csv("data/marketing_campaign.csv", sep='\t')
df = preprocess(df)
df = engineer_features(df)
X_scaled, scaler = scale_features(df)
X_pca, pca = apply_pca(X_scaled)

mlflow.set_experiment("customer-segmentation")

best_score = -1     #silhouette score is always between -1 and 1
best_model = None

# Try different cluster counts
for n in [2, 3, 4, 5]:
    for init in ['k-means++','random']:
        for max_iter in [100, 200, 300]:
            with mlflow.start_run(run_name=f"kmeans-k{n}-{init}-iter{max_iter}"): 
                model = KMeans(n_clusters=n, random_state=42, n_init=10, max_iter=max_iter, init=init)
                labels = model.fit_predict(X_pca)

                score = silhouette_score(X_pca, labels)

                if score > best_score:
                    best_score = score
                    best_model = model  # ← automatically updated!



                mlflow.log_param("n_clusters", n)
                mlflow.log_param("random_state", 42)
                mlflow.log_param("init",init)
                mlflow.log_param("max_iter",max_iter)

                mlflow.log_metric("silhouette_score", round(score, 4))
                mlflow.log_metric("inertia", round(model.inertia_, 2))

                mlflow.sklearn.log_model(model, "kmeans_model")#artifact_path='kmeans_model'

                print(f"k={n} → silhouette={score:.4f}, inertia={model.inertia_:.2f}")

for n in [2,3,4,5]:
    for linkage in ['ward', 'complete', 'average']:
        with mlflow.start_run(run_name=f"AgglomerativeClustering-k{n}-linkage{linkage}"):
            model = AgglomerativeClustering(n_clusters=n, linkage=linkage)
            labels = model.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)

            if score > best_score :
                best_score = score
                best_model = model

            mlflow.log_param("n_clusters", n)
            mlflow.log_param("linkage",linkage)
            mlflow.log_metric("silhouette_score", round(score, 5))
            
            mlflow.sklearn.log_model(model,"AgglomerativeClustering")

            print(f"k={n} → silhouette={score:.4f}")


print("All runs logged!")

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
print(f"Best score: {best_score}")



# best for when using port is used , mlflow server --host 127.0.0.1 --port 5000
# model = mlflow.pyfunc.load_model("models:/customer-segmentation/champion")
# mlflow models serve -m "models:/customer-segmentation/champion" -p 5001, This starts a REST API server that serves your model.

