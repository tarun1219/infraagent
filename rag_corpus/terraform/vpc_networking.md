# Terraform VPC Networking Patterns

## Overview
A well-designed VPC is the foundation of AWS security. Always use private subnets for workloads, NAT Gateways for outbound internet, and NACLs as a secondary defense layer.

## Complete VPC with Public and Private Subnets

```hcl
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "main-vpc"
    Environment = var.environment
  }
}

# Public subnets (for ALBs, NAT Gateways)
resource "aws_subnet" "public" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index)
  availability_zone = var.availability_zones[count.index]

  map_public_ip_on_launch = false    # Never auto-assign public IPs

  tags = {
    Name                     = "public-${var.availability_zones[count.index]}"
    "kubernetes.io/role/elb" = "1"   # For EKS ALB controller
  }
}

# Private subnets (for EC2, EKS, RDS)
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index + 10)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name                              = "private-${var.availability_zones[count.index]}"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# Database subnets (isolated, no route to internet)
resource "aws_subnet" "database" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index + 20)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "database-${var.availability_zones[count.index]}"
  }
}
```

## Internet Gateway and NAT Gateway

```hcl
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "main-igw" }
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat" {
  count  = length(var.availability_zones)
  domain = "vpc"
  tags   = { Name = "nat-eip-${count.index}" }
}

# NAT Gateway per AZ for high availability
resource "aws_nat_gateway" "main" {
  count         = length(var.availability_zones)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  tags          = { Name = "nat-gw-${count.index}" }
  depends_on    = [aws_internet_gateway.main]
}
```

## Route Tables

```hcl
# Public route table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Private route table (one per AZ, routes through NAT Gateway in same AZ)
resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = { Name = "private-rt-${count.index}" }
}

resource "aws_route_table_association" "private" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}
```

## VPC Flow Logs

```hcl
resource "aws_flow_log" "main" {
  vpc_id          = aws_vpc.main.id
  traffic_type    = "ALL"
  iam_role_arn    = aws_iam_role.flow_log_role.arn
  log_destination = aws_cloudwatch_log_group.flow_logs.arn
}
```

## Security Groups vs NACLs

| Feature | Security Groups | NACLs |
|---------|----------------|-------|
| Level | Instance/ENI | Subnet |
| State | Stateful | Stateless |
| Rules | Allow only | Allow and Deny |
| Evaluation | All rules | Ordered by rule number |

## Common Mistakes
- Single NAT Gateway for all AZs — if the AZ fails, all private subnet outbound traffic stops
- `map_public_ip_on_launch = true` on private subnets — accidentally exposes instances
- Overly permissive security groups with `0.0.0.0/0` on inbound — restrict to known CIDRs
- Missing VPC Flow Logs — no visibility into network traffic for incident response
