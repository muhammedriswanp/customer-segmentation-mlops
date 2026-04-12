from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        input_path="data/marketing_campaign.csv",
        output_path="outputs/clusters/dataset_with_clusters.csv"
    )