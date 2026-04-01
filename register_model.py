# register_model.py
# Automatically finds the best model from MLflow experiments
# and registers it to MLflow Model Registry with 'champion' alias.
# Run this after log_experiment.py to update the registry.

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Find best run
experiment = client.get_experiment_by_name("customer-segmentation")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.silhouette_score DESC"],
    max_results=1
)

best_run = runs[0]
print(f"Best run ID : {best_run.info.run_id}")
print(f"Best score  : {best_run.data.metrics['silhouette_score']}")
print(f"Params      : {best_run.data.params}")

# Register model
model_uri = f"runs:/{best_run.info.run_id}/AgglomerativeClustering"
result = mlflow.register_model(
    model_uri=model_uri,
    name="customer-segmentation"
)

print(f"\nModel registered!")
print(f"Name   : {result.name}")
print(f"Version: {result.version}")

# ── Set champion alias ──
client.set_registered_model_alias(
    name="customer-segmentation",
    alias="champion",
    version=result.version
)

print(f"Alias 'champion' set on version {result.version} ✅")