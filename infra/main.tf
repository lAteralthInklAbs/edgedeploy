# EdgeDeploy AWS Infrastructure
# Terraform configuration for model deployment pipeline

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "edgedeploy-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "edgedeploy-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "EdgeDeploy"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# S3 Bucket for model artifacts
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${var.environment}"

  tags = {
    Name = "Model Artifacts"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket for drift detection data
resource "aws_s3_bucket" "drift_data" {
  bucket = "${var.project_name}-drift-data-${var.environment}"

  tags = {
    Name = "Drift Detection Data"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "drift_data" {
  bucket = aws_s3_bucket.drift_data.id

  rule {
    id     = "expire-old-data"
    status = "Enabled"

    expiration {
      days = 90
    }

    filter {
      prefix = "reference/"
    }
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.project_name}-sagemaker-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "${var.project_name}-sagemaker-s3-policy"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*",
          aws_s3_bucket.drift_data.arn,
          "${aws_s3_bucket.drift_data.arn}/*"
        ]
      }
    ]
  })
}

# SageMaker Model
resource "aws_sagemaker_model" "edge_model" {
  name               = "${var.project_name}-model-${var.environment}"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = var.inference_image
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/models/latest/model.tar.gz"
    environment = {
      SAGEMAKER_PROGRAM = "inference.py"
    }
  }

  tags = {
    Name = "Edge Deployment Model"
  }
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "edge_endpoint" {
  name = "${var.project_name}-endpoint-config-${var.environment}"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.edge_model.name
    initial_instance_count = var.endpoint_instance_count
    instance_type          = var.endpoint_instance_type

    serverless_config {
      max_concurrency   = 10
      memory_size_in_mb = 2048
    }
  }

  tags = {
    Name = "Edge Endpoint Config"
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "edge_endpoint" {
  name                 = "${var.project_name}-endpoint-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.edge_endpoint.name

  tags = {
    Name = "Edge Deployment Endpoint"
  }
}

# Lambda for drift detection
resource "aws_lambda_function" "drift_detector" {
  function_name = "${var.project_name}-drift-detector-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 1024

  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      S3_BUCKET        = aws_s3_bucket.drift_data.bucket
      PSI_THRESHOLD    = "0.2"
      MMD_THRESHOLD    = "0.1"
      ALERT_SNS_TOPIC  = aws_sns_topic.drift_alerts.arn
    }
  }

  tags = {
    Name = "Drift Detection Lambda"
  }
}

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda"
  output_path = "${path.module}/lambda.zip"
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3_sns" {
  name = "${var.project_name}-lambda-s3-sns-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.drift_data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = [
          aws_sns_topic.drift_alerts.arn
        ]
      }
    ]
  })
}

# SNS Topic for drift alerts
resource "aws_sns_topic" "drift_alerts" {
  name = "${var.project_name}-drift-alerts-${var.environment}"

  tags = {
    Name = "Drift Detection Alerts"
  }
}

# CloudWatch Event Rule for scheduled drift detection
resource "aws_cloudwatch_event_rule" "drift_check_schedule" {
  name                = "${var.project_name}-drift-check-${var.environment}"
  description         = "Trigger drift detection every hour"
  schedule_expression = "rate(1 hour)"

  tags = {
    Name = "Drift Check Schedule"
  }
}

resource "aws_cloudwatch_event_target" "drift_lambda" {
  rule      = aws_cloudwatch_event_rule.drift_check_schedule.name
  target_id = "DriftDetectorLambda"
  arn       = aws_lambda_function.drift_detector.arn
}

resource "aws_lambda_permission" "allow_cloudwatch" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.drift_detector.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.drift_check_schedule.arn
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-dashboard-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "SageMaker Endpoint Invocations"
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          metrics = [
            ["AWS/SageMaker", "Invocations", "EndpointName", aws_sagemaker_endpoint.edge_endpoint.name]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Endpoint Latency"
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          metrics = [
            ["AWS/SageMaker", "ModelLatency", "EndpointName", aws_sagemaker_endpoint.edge_endpoint.name],
            ["AWS/SageMaker", "OverheadLatency", "EndpointName", aws_sagemaker_endpoint.edge_endpoint.name]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Lambda Drift Detection"
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.drift_detector.function_name],
            ["AWS/Lambda", "Errors", "FunctionName", aws_lambda_function.drift_detector.function_name]
          ]
        }
      }
    ]
  })
}

