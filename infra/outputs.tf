# EdgeDeploy Terraform Outputs

output "model_bucket_name" {
  description = "S3 bucket name for model artifacts"
  value       = aws_s3_bucket.models.bucket
}

output "model_bucket_arn" {
  description = "S3 bucket ARN for model artifacts"
  value       = aws_s3_bucket.models.arn
}

output "drift_data_bucket_name" {
  description = "S3 bucket name for drift detection data"
  value       = aws_s3_bucket.drift_data.bucket
}

output "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name"
  value       = aws_sagemaker_endpoint.edge_endpoint.name
}

output "sagemaker_endpoint_arn" {
  description = "SageMaker endpoint ARN"
  value       = aws_sagemaker_endpoint.edge_endpoint.arn
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "drift_detector_lambda_arn" {
  description = "Drift detector Lambda function ARN"
  value       = aws_lambda_function.drift_detector.arn
}

output "drift_alerts_topic_arn" {
  description = "SNS topic ARN for drift alerts"
  value       = aws_sns_topic.drift_alerts.arn
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "model_upload_command" {
  description = "Command to upload model to S3"
  value       = "aws s3 cp model.tar.gz s3://${aws_s3_bucket.models.bucket}/models/latest/model.tar.gz"
}

output "endpoint_invoke_command" {
  description = "Command to invoke the endpoint"
  value       = "aws sagemaker-runtime invoke-endpoint --endpoint-name ${aws_sagemaker_endpoint.edge_endpoint.name} --body '{}' output.json"
}

