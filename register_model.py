import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # ← use database
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

# Register using run artifact URI
model_uri = f"runs:/{best_run.info.run_id}/AgglomerativeClustering"
result = mlflow.register_model(
    model_uri=model_uri,
    name="customer-segmentation"
)

print(f"\nModel registered!")
print(f"Name   : {result.name}")
print(f"Version: {result.version}")

# python -c "
# import mlflow
# mlflow.set_tracking_uri('sqlite:///mlflow.db')
# result = mlflow.register_model(
#     'runs:/01603c5d655f4e78b00890f023b4ac27/kmeans_model',
#     'customer-segmentation'
# )
# print('Version:', result.version)