# Terraform KMS Encryption Patterns

## Overview
AWS KMS provides centralized key management for encrypting data at rest across S3, RDS, EBS, Secrets Manager, CloudWatch, and other services. Customer-managed keys (CMKs) provide audit trails, rotation control, and fine-grained access.

## KMS Key with Key Policy

```hcl
data "aws_caller_identity" "current" {}

resource "aws_kms_key" "app" {
  description             = "KMS key for my-app encryption"
  deletion_window_in_days = 30          # 7-30 days grace period
  enable_key_rotation     = true        # Annual automatic rotation
  multi_region            = false       # Use multi_region = true for cross-region replication

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow app role to use the key"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.app_role.arn
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },
      {
        Sid    = "Allow CloudWatch to use key for log encryption"
        Effect = "Allow"
        Principal = {
          Service = "logs.${var.aws_region}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*"
          }
        }
      }
    ]
  })

  tags = {
    Name        = "app-kms-key"
    Environment = var.environment
  }
}

resource "aws_kms_alias" "app" {
  name          = "alias/my-app-${var.environment}"
  target_key_id = aws_kms_key.app.key_id
}
```

## S3 Encryption with KMS

```hcl
resource "aws_s3_bucket_server_side_encryption_configuration" "app" {
  bucket = aws_s3_bucket.app.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.app.arn
    }
    bucket_key_enabled = true    # Reduces KMS API call volume and cost
  }
}
```

## RDS with KMS

```hcl
resource "aws_db_instance" "postgres" {
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn
  # Note: kms_key_id must be set BEFORE instance creation; cannot be changed after
}
```

## EBS Volume Encryption with KMS

```hcl
resource "aws_ebs_volume" "data" {
  availability_zone = data.aws_availability_zones.available.names[0]
  size              = 100
  type              = "gp3"
  encrypted         = true
  kms_key_id        = aws_kms_key.ebs.arn
}

# Account-level default encryption (belt-and-suspenders)
resource "aws_ebs_encryption_by_default" "main" {
  enabled = true
}
```

## Secrets Manager with KMS

```hcl
resource "aws_secretsmanager_secret" "db" {
  name                    = "my-app/database"
  kms_key_id              = aws_kms_key.secrets.arn
  recovery_window_in_days = 30

  rotation_rules {
    automatically_after_days = 30
  }
}
```

## Common Mistakes
- Using the default `aws/s3` key — it is account-level, not auditable per resource, and not shareable across accounts
- `deletion_window_in_days = 7` — minimum allowed, but gives only 7 days to recover from accidental key deletion
- `enable_key_rotation = false` — disabling rotation means the same key material is used indefinitely
- No key policy restricting access — without explicit `Principal` restrictions, any IAM entity in the account can use the key
- Forgetting `bucket_key_enabled` — every S3 PUT makes a separate KMS API call without it, increasing costs significantly
