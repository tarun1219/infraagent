# Terraform EC2 Security Best Practices

## Overview
EC2 security involves security groups (firewall), IMDSv2 (metadata service hardening), EBS encryption, IAM instance profiles, and SSH key management.

## EC2 Instance with Security Hardening

```hcl
resource "aws_instance" "app" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = "t3.medium"
  subnet_id     = aws_subnet.private[0].id    # Always use private subnets

  # IAM instance profile (no hardcoded credentials)
  iam_instance_profile = aws_iam_instance_profile.app.name

  # Security group
  vpc_security_group_ids = [aws_security_group.app.id]

  # IMDSv2 enforcement (prevents SSRF-based credential theft)
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"    # IMDSv2 — require signed tokens
    http_put_response_hop_limit = 1             # Prevents container SSRF
    instance_metadata_tags      = "enabled"
  }

  # EBS root volume encryption
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 50
    encrypted             = true
    kms_key_id            = aws_kms_key.ebs.arn
    delete_on_termination = true
  }

  # No public IP (use SSM Session Manager or bastion for access)
  associate_public_ip_address = false

  # Use Systems Manager instead of SSH
  # No key_name — avoids SSH key management

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    environment = var.environment
  }))

  tags = {
    Name        = "app-server"
    Environment = var.environment
    Patch       = "auto"    # Tag for AWS Systems Manager Patch Manager
  }
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}
```

## Security Group (Least Privilege)

```hcl
resource "aws_security_group" "app" {
  name        = "app-sg-${var.environment}"
  description = "Application server security group"
  vpc_id      = aws_vpc.main.id

  # Ingress only from ALB
  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Allow traffic from ALB"
  }

  # Egress to database
  egress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.rds.id]
    description     = "PostgreSQL to RDS"
  }

  # Egress HTTPS to AWS APIs
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS to AWS APIs and internet"
  }

  tags = { Name = "app-sg" }
}
```

## IAM Instance Profile

```hcl
resource "aws_iam_instance_profile" "app" {
  name = "app-instance-profile"
  role = aws_iam_role.app_ec2.name
}

resource "aws_iam_role" "app_ec2" {
  name = "app-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Allow SSM for remote access (replaces SSH)
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
```

## Common Mistakes
- `http_tokens = "optional"` — leaves IMDSv2 opt-in, allowing SSRF attacks to steal instance credentials
- `associate_public_ip_address = true` on instances — exposes the instance directly to the internet
- Inbound `0.0.0.0/0` on port 22 — SSH should never be open to the internet; use SSM Session Manager
- Missing `kms_key_id` on EBS volumes — encrypted flag alone uses the default aws/ebs key which is account-level, not per-resource
- Hardcoded access keys in user_data — these are visible in the EC2 console and logs; use instance profiles
