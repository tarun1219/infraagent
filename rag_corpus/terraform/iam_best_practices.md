# Terraform IAM Best Practices

## Least Privilege Principle

Never use `Action: "*"` or `Resource: "*"`. Always specify exact actions and resources.

```hcl
# CORRECT: least-privilege policy
resource "aws_iam_policy" "app_policy" {
  name        = "app-read-policy"
  description = "Allow app to read from specific S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.app.arn,
          "${aws_s3_bucket.app.arn}/*"
        ]
      }
    ]
  })
}
```

## Common Checkov Failures

| Check ID | Description | Fix |
|----------|-------------|-----|
| CKV_AWS_40 | IAM policy with wildcard action | Replace `*` with specific actions |
| CKV_AWS_53 | S3 bucket policy allows public access | Add Principal condition |
| CKV_AWS_60 | IAM role allows assume from all principals | Add `Condition` block |
| CKV2_AWS_62 | S3 no event notification | Add `aws_s3_bucket_notification` |

## S3 Secure Configuration

```hcl
resource "aws_s3_bucket" "secure" {
  bucket = "my-secure-bucket"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "secure" {
  bucket = aws_s3_bucket.secure.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "secure" {
  bucket                  = aws_s3_bucket.secure.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "secure" {
  bucket = aws_s3_bucket.secure.id
  versioning_configuration {
    status = "Enabled"
  }
}
```
