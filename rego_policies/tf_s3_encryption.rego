package main

# S3 buckets must enable server-side encryption.
# Unencrypted S3 buckets expose data at rest to anyone who gains access to the
# underlying storage. AWS recommends SSE-S3 at minimum; SSE-KMS is preferred
# for auditable key usage.

deny[msg] {
    resource := input.resource.aws_s3_bucket[name]
    not has_encryption(name)
    msg := sprintf("S3 bucket '%v' does not have server-side encryption configured — add aws_s3_bucket_server_side_encryption_configuration", [name])
}

deny[msg] {
    resource := input.resource.aws_s3_bucket_server_side_encryption_configuration[name]
    rule := resource.rule[_]
    apply := rule.apply_server_side_encryption_by_default[_]
    not apply.sse_algorithm
    msg := sprintf("S3 encryption configuration '%v' is missing sse_algorithm (use 'aws:kms' or 'AES256')", [name])
}

has_encryption(bucket_name) {
    input.resource.aws_s3_bucket_server_side_encryption_configuration[_].bucket == bucket_name
}

has_encryption(bucket_name) {
    input.resource.aws_s3_bucket_server_side_encryption_configuration[_].bucket == sprintf("aws_s3_bucket.%v.id", [bucket_name])
}
