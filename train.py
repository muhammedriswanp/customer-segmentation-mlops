import os
from src.pipeline import run_pipeline

if __name__ == "__main__":
    input_dir = os.environ.get("SM_CHANNEL_TRAIN", "data")
    model_dir = os.environ.get("SM_MODEL_DIR", "models")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "outputs/clusters")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    run_pipeline(
        input_path=os.path.join(input_dir, "marketing_campaign.csv"),
        output_path=os.path.join(output_dir, "dataset_with_clusters.csv"),
        model_dir=model_dir
    )