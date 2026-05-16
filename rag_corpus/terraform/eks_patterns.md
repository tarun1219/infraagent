# Terraform EKS Cluster Patterns

## Overview
EKS is the managed Kubernetes service on AWS. A production-ready cluster needs private API server access, managed node groups, IRSA for pod-level AWS access, and proper security group configuration.

## EKS Cluster

```hcl
resource "aws_eks_cluster" "main" {
  name     = "my-cluster-${var.environment}"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.29"

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    security_group_ids      = [aws_security_group.eks_cluster.id]
    endpoint_private_access = true     # Enable private API endpoint
    endpoint_public_access  = false    # Disable public API endpoint in production
    # If public is needed: restrict to known CIDRs
    # public_access_cidrs = ["203.0.113.0/24"]
  }

  encryption_config {
    resources = ["secrets"]
    provider {
      key_arn = aws_kms_key.eks.arn
    }
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  kubernetes_network_config {
    ip_family         = "ipv4"
    service_cidr      = "172.20.0.0/16"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_cloudwatch_log_group.eks,
  ]

  tags = {
    Name        = "my-cluster"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/my-cluster-${var.environment}/cluster"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.cloudwatch.arn
}
```

## EKS Managed Node Group

```hcl
resource "aws_eks_node_group" "app" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "app-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id    # Worker nodes in private subnets

  ami_type       = "AL2_x86_64"
  instance_types = ["m5.large"]
  disk_size      = 50

  scaling_config {
    desired_size = 3
    min_size     = 2
    max_size     = 10
  }

  update_config {
    max_unavailable_percentage = 25    # Allows rolling updates with minimal disruption
  }

  # IMDSv2 enforcement on nodes
  launch_template {
    id      = aws_launch_template.eks_node.id
    version = aws_launch_template.eks_node.latest_version
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_ecr_readonly,
  ]

  tags = {
    "k8s.io/cluster-autoscaler/enabled"              = "true"
    "k8s.io/cluster-autoscaler/my-cluster-${var.environment}" = "owned"
  }
}

resource "aws_launch_template" "eks_node" {
  name_prefix = "eks-node-"

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"    # IMDSv2
    http_put_response_hop_limit = 2             # 2 hops needed for containers on nodes
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 50
      volume_type           = "gp3"
      encrypted             = true
      kms_key_id            = aws_kms_key.ebs.arn
      delete_on_termination = true
    }
  }
}
```

## IRSA Setup (OIDC Provider)

```hcl
data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
}
```

## Common Mistakes
- `endpoint_public_access = true` without `public_access_cidrs` — exposes the API server to the entire internet
- Worker nodes in public subnets — nodes should be in private subnets; only LBs go in public subnets
- `http_tokens = "optional"` in node launch template — enables SSRF-based instance credential theft from pods
- Missing `encryption_config` for secrets — Kubernetes secrets (including service account tokens) are unencrypted in etcd without this
- `retention_in_days = 0` on control plane logs — audit logs are needed for incident investigation
