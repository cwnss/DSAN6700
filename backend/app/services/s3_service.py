import boto3
from botocore.client import Config
from backend.app.config import settings
print("DEBUG BOTO CREDS:", settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY)

s3 = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)


async def upload_file_to_s3(file, bucket: str, key: str) -> str:
    # MUST use the already authenticated client
    body = await file.read()

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=file.content_type,
    )

    return f"s3://{bucket}/{key}"
