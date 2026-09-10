import boto3

bucket_name = "sales-demand-forecasting-pipeline-2026"
local_file = "output/forecast.csv"
object_name = "predictions/forecast.csv"

s3 = boto3.client("s3")
s3.upload_file(local_file, bucket_name, object_name)

print("Forecast uploaded successfully!")