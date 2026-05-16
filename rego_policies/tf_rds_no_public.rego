package main

# RDS instances must not be publicly accessible.
# Setting publicly_accessible = true exposes the database endpoint on the
# internet, dramatically increasing the attack surface. RDS should be placed
# in private subnets and accessed via application-layer connections only.

deny[msg] {
    db := input.resource.aws_db_instance[name]
    db.publicly_accessible == true
    msg := sprintf("RDS instance '%v' has publicly_accessible = true — place it in a private subnet", [name])
}

deny[msg] {
    db := input.resource.aws_db_instance[name]
    not db.storage_encrypted
    msg := sprintf("RDS instance '%v' does not enable storage_encrypted — set storage_encrypted = true", [name])
}

deny[msg] {
    db := input.resource.aws_db_instance[name]
    not db.multi_az
    msg := sprintf("RDS instance '%v' does not enable multi_az — set multi_az = true for production HA", [name])
}

deny[msg] {
    db := input.resource.aws_db_instance[name]
    not db.deletion_protection
    msg := sprintf("RDS instance '%v' does not enable deletion_protection — set deletion_protection = true", [name])
}

deny[msg] {
    cluster := input.resource.aws_rds_cluster[name]
    cluster.publicly_accessible == true
    msg := sprintf("RDS cluster '%v' has publicly_accessible = true — restrict to private subnets", [name])
}
