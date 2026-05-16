# Terraform IAM Patterns and Least Privilege

## Overview
IAM is the most security-critical service in AWS. Follow least privilege: grant only the permissions needed for the specific task, to specific resources, with conditions where possible.

## IAM Role with Trust Policy

```hcl
resource "aws_iam_role" "app_role" {
  name                 = "my-app-role-${var.environment}"
  max_session_duration = 3600    # 1 hour max

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.aws_region    # Restrict to one region
          }
        }
      }
    ]
  })

  tags = {
    Name        = "my-app-role"
    Environment = var.environment
  }
}
```

## Least-Privilege IAM Policy

```hcl
resource "aws_iam_policy" "app_policy" {
  name        = "my-app-policy"
  description = "Allows my-app to read from S3 and write to DynamoDB"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ReadAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
      },
      {
        Sid    = "DynamoDBWrite"
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:GetItem",
          "dynamodb:Query"
        ]
        Resource = aws_dynamodb_table.app_table.arn
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.app_key.arn
        Condition = {
          StringEquals = {
            "kms:ViaService" = "s3.${var.aws_region}.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "app" {
  role       = aws_iam_role.app_role.name
  policy_arn = aws_iam_policy.app_policy.arn
}
```

## EKS IRSA (IAM Roles for Service Accounts)

```hcl
data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

data "aws_iam_openid_connect_provider" "eks" {
  url = data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer
}

resource "aws_iam_role" "irsa_role" {
  name = "my-app-irsa-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = data.aws_iam_openid_connect_provider.eks.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(data.aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub" = "system:serviceaccount:${var.namespace}:${var.service_account_name}"
            "${replace(data.aws_iam_openid_connect_provider.eks.url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
}
```

## Permission Boundary

```hcl
resource "aws_iam_policy" "boundary" {
  name = "developer-permission-boundary"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:*", "dynamodb:*", "lambda:*"]
        Resource = "*"
      },
      {
        Effect   = "Deny"
        Action   = ["iam:CreateUser", "iam:DeleteUser", "organizations:*"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role" "developer_role" {
  name                 = "developer-role"
  permissions_boundary = aws_iam_policy.boundary.arn
  assume_role_policy   = data.aws_iam_policy_document.assume.json
}
```

## Common Mistakes
- Using `Action = "*"` or `Resource = "*"` without specific conditions
- Attaching `AdministratorAccess` managed policy to application roles
- No conditions on cross-account trust relationships — allows any principal in the trusted account
- Using long-lived IAM users instead of roles — prefer roles with temporary credentials
- Not rotating access keys — use AWS Secrets Manager or enforce 90-day rotation
