import boto3
from botocore.client import Config
from backend.app.config import settings
print("DEBUG BOTO CREDS:", settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY)

# Use Signature Version 4 for presigned URLs (required by modern S3)
s3 = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')  # Force Signature Version 4
)


async def upload_file_to_s3(file, bucket: str, key: str) -> str:
    file.seek(0)
    body = file.read()

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="image/jpeg",
    )

    return f"s3://{bucket}/{key}"


def get_presigned_url(s3_uri: str, expiration: int = 3600) -> str:
    """
    Convert s3://bucket/key to a presigned HTTPS URL that can be accessed publicly.
    expiration: URL validity in seconds (default 1 hour)
    """
    if not s3_uri or not s3_uri.startswith("s3://"):
        return s3_uri  # Return as-is if not an S3 URI
    
    # Parse s3://bucket/key
    parts = s3_uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        return s3_uri  # Invalid format, return as-is
    
    bucket = parts[0]
    key = parts[1]
    
    try:
        # Detect the bucket's actual region (bucket might be in different region than configured)
        try:
            bucket_region = s3.get_bucket_location(Bucket=bucket)['LocationConstraint']
            # us-east-1 returns None, so handle that
            if bucket_region is None:
                bucket_region = 'us-east-1'
        except Exception:
            # If detection fails, use configured region
            bucket_region = settings.AWS_REGION
        
        # Create a client with the correct region for this bucket
        s3_client = boto3.client(
            "s3",
            region_name=bucket_region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4')
        )
        
        # Generate presigned URL with the correct region
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        # If presigned URL generation fails, return original
        print(f"Warning: Failed to generate presigned URL for {s3_uri}: {e}")
        return s3_uri
