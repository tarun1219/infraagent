# Terraform Lambda Function Patterns

## Overview
Lambda is AWS's serverless compute. Best practices cover IAM (least-privilege execution role), environment variables (use Secrets Manager, not plaintext), VPC configuration, and observability.

## Lambda Function with IAM Role

```hcl
resource "aws_iam_role" "lambda_exec" {
  name = "my-lambda-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
  # Use AWSLambdaVPCAccessExecutionRole if the function is in a VPC
  # Use AWSLambdaBasicExecutionRole if not in VPC
}

resource "aws_iam_role_policy" "lambda_custom" {
  name = "lambda-custom-policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.db_creds.arn]
      },
      {
        Effect = "Allow"
        Action = ["kms:Decrypt"]
        Resource = [aws_kms_key.lambda.arn]
      }
    ]
  })
}
```

## Lambda Function Resource

```hcl
resource "aws_lambda_function" "my_function" {
  function_name = "my-function-${var.environment}"
  role          = aws_iam_role.lambda_exec.arn

  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  handler          = "index.handler"
  runtime          = "nodejs20.x"

  timeout     = 30      # seconds
  memory_size = 256     # MB

  # KMS encryption for environment variables
  kms_key_arn = aws_kms_key.lambda.arn

  environment {
    variables = {
      ENVIRONMENT   = var.environment
      SECRET_ARN    = aws_secretsmanager_secret.db_creds.arn
      # Never put actual secrets here — reference Secrets Manager ARN instead
    }
  }

  # VPC configuration (if Lambda needs to reach RDS or other VPC resources)
  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }

  # Dead letter queue for failed async invocations
  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  # X-Ray tracing
  tracing_config {
    mode = "Active"
  }

  # Reserved concurrency (optional — prevents runaway scaling)
  reserved_concurrent_executions = 100

  layers = [aws_lambda_layer_version.dependencies.arn]

  tags = {
    Name        = "my-function"
    Environment = var.environment
  }
}

# CloudWatch log group with retention
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/my-function-${var.environment}"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.cloudwatch.arn
}
```

## Lambda Security Group

```hcl
resource "aws_security_group" "lambda" {
  name        = "lambda-sg"
  description = "Lambda function security group"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]    # HTTPS to AWS APIs and internet
  }

  egress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.rds.id]
  }
}
```

## Common Mistakes
- Storing secrets in environment variables as plaintext — use Secrets Manager and fetch at runtime
- No VPC config when Lambda needs to reach RDS — Lambda in VPC requires VPC endpoints or NAT Gateway for public AWS API calls
- Forgetting CloudWatch log group resource — Lambda creates it automatically but without retention or KMS encryption
- No DLQ for async invocations — failed events are silently dropped after 2 retries
- `timeout = 900` (max) for all functions — set realistic timeouts to prevent runaway execution costs
