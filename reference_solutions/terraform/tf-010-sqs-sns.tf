resource "aws_kms_key" "messaging" {
  description             = "KMS key for SQS/SNS encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  tags = { ManagedBy = "terraform" }
}

resource "aws_sqs_queue" "main" {
  name                       = "main-queue"
  message_retention_seconds  = 86400
  visibility_timeout_seconds = 30
  kms_master_key_id          = aws_kms_key.messaging.id

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })

  tags = {
    Name      = "main-queue"
    ManagedBy = "terraform"
  }
}

resource "aws_sqs_queue" "dlq" {
  name                      = "main-queue-dlq"
  message_retention_seconds = 1209600
  kms_master_key_id         = aws_kms_key.messaging.id

  tags = {
    Name      = "main-queue-dlq"
    ManagedBy = "terraform"
  }
}

resource "aws_sqs_queue_policy" "main" {
  queue_url = aws_sqs_queue.main.id
  policy    = data.aws_iam_policy_document.sqs.json
}

data "aws_iam_policy_document" "sqs" {
  statement {
    sid    = "AllowSNS"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["sns.amazonaws.com"]
    }
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.main.arn]
    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [aws_sns_topic.main.arn]
    }
  }
}

resource "aws_sns_topic" "main" {
  name              = "main-topic"
  kms_master_key_id = aws_kms_key.messaging.id

  tags = {
    Name      = "main-topic"
    ManagedBy = "terraform"
  }
}

resource "aws_sns_topic_subscription" "sqs" {
  topic_arn = aws_sns_topic.main.arn
  protocol  = "sqs"
  endpoint  = aws_sqs_queue.main.arn
}
