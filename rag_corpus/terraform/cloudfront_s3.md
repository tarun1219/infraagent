# Terraform CloudFront + S3 Static Site Patterns

## Overview
CloudFront serves as a CDN and security layer in front of S3. Use Origin Access Control (OAC, the modern replacement for OAI) to ensure S3 is only accessible through CloudFront, never directly.

## CloudFront with OAC (Recommended — OAI is legacy)

```hcl
resource "aws_s3_bucket" "static_site" {
  bucket = "my-static-site-${var.environment}"
}

resource "aws_s3_bucket_public_access_block" "static_site" {
  bucket                  = aws_s3_bucket.static_site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudfront_origin_access_control" "main" {
  name                              = "s3-oac"
  description                       = "OAC for static site S3 bucket"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  aliases             = ["www.example.com"]
  price_class         = "PriceClass_100"    # US and Europe only

  origin {
    domain_name              = aws_s3_bucket.static_site.bucket_regional_domain_name
    origin_id                = "s3-static-site"
    origin_access_control_id = aws_cloudfront_origin_access_control.main.id
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "s3-static-site"
    viewer_protocol_policy = "redirect-to-https"    # Always enforce HTTPS
    compress               = true

    cache_policy_id            = data.aws_cloudfront_cache_policy.caching_optimized.id
    origin_request_policy_id   = data.aws_cloudfront_origin_request_policy.s3_cors.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id
  }

  # Custom error pages for SPA routing
  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.main.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"    # Modern TLS only
  }

  web_acl_id = aws_wafv2_web_acl.main.arn    # WAF for DDoS and bot protection

  logging_config {
    bucket          = aws_s3_bucket.cf_logs.bucket_domain_name
    include_cookies = false
    prefix          = "cloudfront/"
  }
}
```

## Security Headers Response Policy

```hcl
resource "aws_cloudfront_response_headers_policy" "security_headers" {
  name = "security-headers-policy"

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains         = true
      preload                    = true
      override                   = true
    }
    content_type_options {
      override = true
    }
    frame_options {
      frame_option = "DENY"
      override     = true
    }
    xss_protection {
      mode_block = true
      protection = true
      override   = true
    }
    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }
  }
}
```

## S3 Bucket Policy (Allow Only CloudFront OAC)

```hcl
resource "aws_s3_bucket_policy" "static_site" {
  bucket = aws_s3_bucket.static_site.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "cloudfront.amazonaws.com" }
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.static_site.arn}/*"
      Condition = {
        StringEquals = {
          "AWS:SourceArn" = aws_cloudfront_distribution.main.arn
        }
      }
    }]
  })
}
```

## Common Mistakes
- Using OAI (Origin Access Identity) — OAC is the modern replacement with better security
- `viewer_protocol_policy = "allow-all"` — serves content over HTTP, enabling MITM attacks
- `minimum_protocol_version = "TLSv1"` — TLS 1.0/1.1 are deprecated; use `TLSv1.2_2021`
- No WAF association — CloudFront without WAF is vulnerable to DDoS and bot abuse
- No response headers policy — missing HSTS, X-Frame-Options, CSP headers
