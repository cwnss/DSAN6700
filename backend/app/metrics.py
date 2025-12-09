"""
Custom Prometheus metrics for ML inference, AWS operations, etc.
"""
from prometheus_client import Counter, Histogram

# ML Inference metrics
ML_INFERENCE_TIME = Histogram(
    "ml_inference_seconds",
    "Time spent on ML inference operations",
    ["operation"],  # embedding, classification, recommendation
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# AWS S3 metrics
AWS_S3_CALLS = Counter(
    "aws_s3_calls_total",
    "Total number of AWS S3 API calls",
    ["operation"]  # upload, get_presigned_url
)

AWS_S3_BYTES = Counter(
    "aws_s3_bytes_total",
    "Total bytes transferred to/from S3",
    ["direction"]  # upload, download
)

AWS_S3_TIME = Histogram(
    "aws_s3_operation_seconds",
    "Time spent on S3 operations",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
