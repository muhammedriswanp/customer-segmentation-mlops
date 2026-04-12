import pandas as pd
import os
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features, scale_features
from src.clustering import apply_pca
import joblib


def run_pipeline(input_path, output_path):
    print("Loading Data...")
    df = pd.read_csv(input_path, sep='\t')

    # Create output folders if they don't exist
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/clusters', exist_ok=True)

    print("Preprocessing...")
    df = preprocess(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Scaling features ...")
    X_scaled, scaler = scale_features(df)

    print("Applying PCA...")
    X_pca, pca = apply_pca(X_scaled)

    print("Clustering...")
    model = joblib.load('models/best_model.pkl')
    labels = model.fit_predict(X_pca)

    df['Cluster'] = labels

    print("Saving models...")
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, "models/pca.pkl")
    joblib.dump(model, "models/kmeans_model.pkl")

    print("Saving results...")
    df.to_csv(output_path, index=False)

    print("Pipeline complete!")
    return df