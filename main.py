from src.pipeline import run_pipeline

df = run_pipeline(
    input_path='data/raw/marketing_campaign.csv',
    output_path='outputs/clusters/dataset_with_clusters.csv'
)