import boto3
import os

bucket_name = "sales-demand-forecasting-pipeline-2026"
object_name = "raw-data/sales.csv"
local_file = "data/raw/sales.csv"

# Create local folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

s3 = boto3.client("s3")
s3.download_file(bucket_name, object_name, local_file)

print("✅ Dataset downloaded successfully.")