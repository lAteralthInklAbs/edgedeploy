# EdgeDeploy Terraform Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "edgedeploy"
}

variable "inference_image" {
  description = "Docker image URI for SageMaker inference"
  type        = string
  default     = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310"
}

variable "endpoint_instance_type" {
  description = "SageMaker endpoint instance type"
  type        = string
  default     = "ml.m5.large"
}

variable "endpoint_instance_count" {
  description = "Number of endpoint instances"
  type        = number
  default     = 1
}

variable "drift_check_schedule" {
  description = "Schedule expression for drift detection (cron or rate)"
  type        = string
  default     = "rate(1 hour)"
}

variable "psi_threshold" {
  description = "PSI threshold for drift detection"
  type        = number
  default     = 0.2
}

variable "mmd_threshold" {
  description = "MMD threshold for drift detection"
  type        = number
  default     = 0.1
}

variable "alert_email" {
  description = "Email address for drift alerts"
  type        = string
  default     = ""
}

variable "enable_canary" {
  description = "Enable canary deployment endpoint"
  type        = bool
  default     = true
}

variable "canary_traffic_percentage" {
  description = "Initial canary traffic percentage"
  type        = number
  default     = 5

  validation {
    condition     = var.canary_traffic_percentage >= 0 && var.canary_traffic_percentage <= 100
    error_message = "Canary traffic percentage must be between 0 and 100."
  }
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

