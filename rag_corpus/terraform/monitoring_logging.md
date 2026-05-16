# Terraform Monitoring and Logging Patterns

## Overview
A complete observability stack on AWS includes CloudWatch for metrics/logs, CloudTrail for API audit, Config for compliance, and GuardDuty for threat detection.

## CloudTrail (API Audit Logging)

```hcl
resource "aws_cloudtrail" "main" {
  name                          = "org-trail"
  s3_bucket_name                = aws_s3_bucket.cloudtrail.id
  include_global_service_events = true    # IAM, STS, CloudFront events
  is_multi_region_trail         = true    # All regions
  enable_log_file_validation    = true    # Detects log tampering
  kms_key_id                    = aws_kms_key.cloudtrail.arn

  event_selector {
    read_write_type           = "All"
    include_management_events = true

    data_resource {
      type   = "AWS::S3::Object"
      values = ["arn:aws:s3:::"]    # All S3 objects (data events)
    }

    data_resource {
      type   = "AWS::Lambda::Function"
      values = ["arn:aws:lambda"]    # All Lambda invocations
    }
  }

  cloud_watch_logs_group_arn = "${aws_cloudwatch_log_group.cloudtrail.arn}:*"
  cloud_watch_logs_role_arn  = aws_iam_role.cloudtrail_cloudwatch.arn

  tags = { Name = "org-trail" }
}
```

## CloudWatch Alarms (Key Security Alarms)

```hcl
# Alarm on root account login
resource "aws_cloudwatch_metric_alarm" "root_login" {
  alarm_name          = "root-account-login"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "RootAccountUsage"
  namespace           = "CloudTrailMetrics"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  alarm_description   = "Alert on any root account login"
}

# Alarm on IAM policy changes
resource "aws_cloudwatch_metric_alarm" "iam_changes" {
  alarm_name          = "iam-policy-changes"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "IAMPolicyChanges"
  namespace           = "CloudTrailMetrics"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
}

# High 5xx error rate on ALB
resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "alb-5xx-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 10
  alarm_actions       = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "error_rate"
    expression  = "errors/requests*100"
    label       = "5XX Error Rate %"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      period      = 60
      stat        = "Sum"
      dimensions = { LoadBalancer = aws_lb.main.arn_suffix }
    }
  }

  metric_query {
    id = "requests"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "RequestCount"
      period      = 60
      stat        = "Sum"
      dimensions = { LoadBalancer = aws_lb.main.arn_suffix }
    }
  }
}
```

## GuardDuty

```hcl
resource "aws_guardduty_detector" "main" {
  enable = true

  datasources {
    s3_logs {
      enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
    malware_protection {
      scan_ec2_instance_with_findings {
        ebs_volumes {
          enable = true
        }
      }
    }
  }

  tags = { Name = "guardduty-detector" }
}
```

## AWS Config

```hcl
resource "aws_config_configuration_recorder" "main" {
  name     = "main-recorder"
  role_arn = aws_iam_role.config.arn

  recording_group {
    all_supported                 = true
    include_global_resource_types = true
  }
}

resource "aws_config_rule" "s3_public_access" {
  name = "s3-bucket-public-access-prohibited"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_PUBLIC_READ_PROHIBITED"
  }

  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_rule" "encrypted_volumes" {
  name = "encrypted-volumes"

  source {
    owner             = "AWS"
    source_identifier = "ENCRYPTED_VOLUMES"
  }
}
```

## Common Mistakes
- `is_multi_region_trail = false` — regional trails miss IAM and other global events
- `enable_log_file_validation = false` — allows attackers to tamper with CloudTrail logs without detection
- GuardDuty disabled — many organizations enable it only after an incident; enable it proactively
- No CloudWatch alarms on CloudTrail metrics — logs are useless without automated alerting
- CloudTrail S3 bucket without lifecycle rules — audit logs accumulate indefinitely, increasing costs
