import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model("models:/customer-segmentation@champion")

print(f"Model loaded: {type(model).__name__}")
print(f"Clusters   : {model.n_clusters}")
