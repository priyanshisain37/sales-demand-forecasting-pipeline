import boto3

bucket_name = "sales-demand-forecasting-pipeline-2026"
local_file = "data/cleaned/cleaned_sales.csv"
object_name = "cleaned-data/cleaned_sales.csv"

s3 = boto3.client("s3")

s3.upload_file(local_file, bucket_name, object_name)

print("Cleaned dataset uploaded successfully!")