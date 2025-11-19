provider "aws" {
  region = "us-east-1"
}

resource "aws_ecr_repository" "backend" {
  name = "spectrashield-backend"
}

resource "aws_ecr_repository" "ml_engine" {
  name = "spectrashield-ml-engine"
}

resource "aws_ecr_repository" "frontend" {
  name = "spectrashield-frontend"
}

resource "aws_eks_cluster" "main" {
  name     = "spectrashield-cluster"
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids = var.subnet_ids
  }
}

resource "aws_iam_role" "eks_cluster" {
  name = "spectrashield-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      },
    ]
  })
}

variable "subnet_ids" {
  type = list(string)
}
