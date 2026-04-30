import pandas as pd
import os
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features, scale_features
from src.clustering import apply_pca
import joblib

def run_pipeline(input_path, output_path, model_dir="models"):
    print("Loading Data...")
    df = pd.read_csv(input_path, sep='\t')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Preprocessing...")
    df = preprocess(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Scaling features...")
    X_scaled, scaler = scale_features(df)

    print("Applying PCA...")
    X_pca, pca = apply_pca(X_scaled)

    print("Clustering...")
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)

    df['Cluster'] = labels

    print("Saving models...")
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(model_dir, 'pca.pkl'))
    joblib.dump(model, os.path.join(model_dir, 'kmeans_model.pkl'))

    print("Saving results...")
    df.to_csv(output_path, index=False)

    print("Pipeline complete!")
    return df