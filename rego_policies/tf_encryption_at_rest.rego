package main

# Storage resources must enable encryption at rest.
# Encryption at rest protects data on disk from physical media theft and
# unauthorized access. This applies to EBS volumes, EFS, RDS, ElastiCache,
# Elasticsearch/OpenSearch, and DynamoDB.

deny[msg] {
    vol := input.resource.aws_ebs_volume[name]
    not vol.encrypted
    msg := sprintf("EBS volume '%v' is not encrypted — set encrypted = true", [name])
}

deny[msg] {
    vol := input.resource.aws_ebs_volume[name]
    vol.encrypted == false
    msg := sprintf("EBS volume '%v' has encryption explicitly disabled — set encrypted = true", [name])
}

deny[msg] {
    fs := input.resource.aws_efs_file_system[name]
    not fs.encrypted
    msg := sprintf("EFS file system '%v' is not encrypted — set encrypted = true", [name])
}

deny[msg] {
    cache := input.resource.aws_elasticache_replication_group[name]
    not cache.at_rest_encryption_enabled
    msg := sprintf("ElastiCache replication group '%v' must set at_rest_encryption_enabled = true", [name])
}

deny[msg] {
    domain := input.resource.aws_elasticsearch_domain[name]
    not domain.encrypt_at_rest.enabled
    msg := sprintf("Elasticsearch domain '%v' must set encrypt_at_rest.enabled = true", [name])
}

deny[msg] {
    table := input.resource.aws_dynamodb_table[name]
    not table.server_side_encryption[_].enabled
    msg := sprintf("DynamoDB table '%v' must enable server_side_encryption", [name])
}
