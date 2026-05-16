package main

# S3 buckets must block public ACLs and public policies.
# Misconfigured S3 buckets are one of the most common causes of cloud data
# breaches. All four block_public_access settings should be enabled unless
# the bucket is intentionally serving public static content.

deny[msg] {
    input.resource.aws_s3_bucket[name]
    not has_public_access_block(name)
    msg := sprintf("S3 bucket '%v' is missing aws_s3_bucket_public_access_block — public access may be allowed", [name])
}

deny[msg] {
    block := input.resource.aws_s3_bucket_public_access_block[name]
    block.block_public_acls != true
    msg := sprintf("S3 public access block '%v' must set block_public_acls = true", [name])
}

deny[msg] {
    block := input.resource.aws_s3_bucket_public_access_block[name]
    block.block_public_policy != true
    msg := sprintf("S3 public access block '%v' must set block_public_policy = true", [name])
}

deny[msg] {
    block := input.resource.aws_s3_bucket_public_access_block[name]
    block.ignore_public_acls != true
    msg := sprintf("S3 public access block '%v' must set ignore_public_acls = true", [name])
}

deny[msg] {
    block := input.resource.aws_s3_bucket_public_access_block[name]
    block.restrict_public_buckets != true
    msg := sprintf("S3 public access block '%v' must set restrict_public_buckets = true", [name])
}

has_public_access_block(bucket_name) {
    input.resource.aws_s3_bucket_public_access_block[_].bucket == bucket_name
}
