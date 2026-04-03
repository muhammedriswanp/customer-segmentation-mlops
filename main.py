from src.pipeline import run_pipeline
import os

input_file = os.getenv('INPUT_FILE', 'marketing_campaign.csv')  # default fallback

df = run_pipeline(
    input_path=f'data/{input_file}',
    output_path='outputs/clusters/dataset_with_clusters.csv'
)

print("Main + Docker version")