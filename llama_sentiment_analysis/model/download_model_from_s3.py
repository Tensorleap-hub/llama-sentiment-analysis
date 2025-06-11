import boto3
import os

s3 = boto3.client('s3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    # region_name='us-east-1'  # replace as needed
)

bucket_name = 'demo-models-nir'
s3_key = 'llama_32_1b_inst.h5'
download_path = s3_key

s3.download_file(bucket_name, s3_key, download_path)
print("File downloaded successfully.")