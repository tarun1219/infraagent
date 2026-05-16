# Terraform RDS Best Practices

## Overview
RDS requires careful configuration for security, availability, and recoverability. Key requirements: encryption, no public access, Multi-AZ, automated backups, and deletion protection.

## Production RDS PostgreSQL

```hcl
resource "aws_db_subnet_group" "main" {
  name       = "main-db-subnet-group"
  subnet_ids = aws_subnet.database[*].id

  tags = { Name = "main-db-subnet-group" }
}

resource "aws_security_group" "rds" {
  name        = "rds-sg"
  description = "Allow PostgreSQL from app tier only"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]    # Only from app SG
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "postgres" {
  identifier = "my-app-postgres-${var.environment}"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"

  allocated_storage     = 100
  max_allocated_storage = 1000     # Enable autoscaling up to 1TB
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.rds.arn

  db_name  = "appdb"
  username = "appuser"
  password = random_password.rds.result    # Never hardcode passwords

  multi_az               = true      # Standby in another AZ
  publicly_accessible    = false     # NEVER expose to internet
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 14       # 14 days of automated backups
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  deletion_protection       = true   # Prevents accidental deletion
  skip_final_snapshot       = false
  final_snapshot_identifier = "my-app-postgres-final-${formatdate("YYYY-MM-DD", timestamp())}"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  monitoring_interval             = 60     # Enhanced monitoring every 60s
  monitoring_role_arn             = aws_iam_role.rds_monitoring.arn

  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  performance_insights_kms_key_id       = aws_kms_key.rds.arn

  auto_minor_version_upgrade = true    # Apply minor version patches automatically

  parameter_group_name = aws_db_parameter_group.postgres.name

  tags = {
    Name        = "my-app-postgres"
    Environment = var.environment
  }
}

resource "random_password" "rds" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_secretsmanager_secret" "rds_password" {
  name                    = "my-app/rds/password"
  recovery_window_in_days = 30
  kms_key_id              = aws_kms_key.secrets.arn
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  secret_id = aws_secretsmanager_secret.rds_password.id
  secret_string = jsonencode({
    username = aws_db_instance.postgres.username
    password = random_password.rds.result
    host     = aws_db_instance.postgres.address
    port     = aws_db_instance.postgres.port
    dbname   = aws_db_instance.postgres.db_name
  })
}
```

## RDS Parameter Group

```hcl
resource "aws_db_parameter_group" "postgres" {
  name   = "my-app-postgres15"
  family = "postgres15"

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name  = "log_disconnections"
    value = "1"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"    # Log queries taking > 1 second
  }
}
```

## Common Mistakes
- `publicly_accessible = true` — the most dangerous RDS misconfiguration
- `deletion_protection = false` without understanding the risk — one `terraform destroy` deletes the database and all data
- `skip_final_snapshot = true` in production — losing the final snapshot means no recovery after accidental deletion
- `backup_retention_period = 0` — disables automated backups entirely
- Hardcoding passwords in Terraform — use `random_password` + Secrets Manager
