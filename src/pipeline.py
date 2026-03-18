import pandas as pd
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features, scale_features
from src.clustering import apply_pca, run_kmeans
import joblib


def run_pipeline(input_path, output_path):
    print("Loading Data...")
    df = pd.read_csv(input_path, sep='\t')

    print("Preprocessing...")
    df = preprocess(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Scaling features ...")
    X_scaled, scaler = scale_features(df)

    print("Applying PCA...")
    X_pca, pca = apply_pca(X_scaled)

    print("Clustering...")
    model, labels = run_kmeans(X_pca, n_clusters=3)
    df['KMeans_Cluster'] = labels

    print("Saving models...")
    joblib.dump(scaler, 'outputs/models/scaler.pkl')
    joblib.dump(pca, "outputs/models/pca.pkl")
    joblib.dump(model, "outputs/models/kmeans_model.pkl")

    print("Saving results...")
    df.to_csv(output_path, index=False)

    print("Pipeline complete!")
    return df