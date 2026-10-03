import os
import sys
from huggingface_hub import hf_hub_download, HfApi

def download_datasets():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    
    token_path = os.path.expanduser('~/.cache/huggingface/token')
    token = None
    if os.path.exists(token_path):
        token = open(token_path).read().strip()
    
    print("Downloading starter dataset...")
    try:
        fpath = hf_hub_download(
            repo_id="FlyRank/internship-starter",
            filename="content_refresh_anonymized.csv",
            repo_type="dataset",
            local_dir=data_dir,
            token=token
        )
        print(f"Starter dataset saved to: {fpath}")
    except Exception as e:
        print(f"Error fetching starter: {e}")

    print("Downloading lane datasets...")
    for lane in ['ranking_lifecycle', 'decline_recovery', 'engagement_fix', 'ai_opportunity']:
        try:
            fpath = hf_hub_download(
                repo_id="FlyRank/internship-lanes",
                filename=f"default_lanes/{lane}.parquet",
                repo_type="dataset",
                local_dir=data_dir,
                token=token
            )
            print(f"Lane {lane} saved to: {fpath}")
        except Exception as e:
            print(f"Error fetching lane {lane}: {e}")

if __name__ == '__main__':
    download_datasets()
