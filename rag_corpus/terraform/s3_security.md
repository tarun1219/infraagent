# Terraform S3 Bucket Security Best Practices

## Overview
S3 misconfiguration is one of the leading causes of cloud data breaches. Every S3 bucket should have encryption, public access blocking, versioning, and access logging enabled by default.

## Complete Secure S3 Bucket

```hcl
resource "aws_s3_bucket" "app_data" {
  bucket = "my-company-app-data-${var.environment}"

  tags = {
    Name        = "app-data"
    Environment = var.environment
    Owner       = "platform-team"
  }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning for recovery from accidental deletes/overwrites
resource "aws_s3_bucket_versioning" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption with KMS
resource "aws_s3_bucket_server_side_encryption_configuration" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3_key.arn
    }
    bucket_key_enabled = true   # Reduces KMS API call costs
  }
}

# Enable access logging
resource "aws_s3_bucket_logging" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "s3-access-logs/app-data/"
}

# Lifecycle policy to manage costs
resource "aws_s3_bucket_lifecycle_configuration" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# Enforce TLS-only access via bucket policy
resource "aws_s3_bucket_policy" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyNonTLS"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.app_data]
}
```

## KMS Key for S3

```hcl
resource "aws_kms_key" "s3_key" {
  description             = "KMS key for S3 bucket encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Name = "s3-encryption-key"
  }
}

resource "aws_kms_alias" "s3_key" {
  name          = "alias/s3-app-data"
  target_key_id = aws_kms_key.s3_key.key_id
}
```

## Common Mistakes
- Using `sse_algorithm = "AES256"` (SSE-S3) — lacks audit trail; prefer `aws:kms`
- Forgetting `depends_on = [aws_s3_bucket_public_access_block.*]` before bucket policy — policy may fail if public access block isn't applied first
- Using the same bucket for both data and access logs — creates a logging loop
- Not enabling `bucket_key_enabled = true` — each S3 PUT makes a KMS API call; bucket keys batch these and reduce costs by ~99%
