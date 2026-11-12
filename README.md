# Iris API on AWS

This AWS-native pipeline uses GitHub as the source repository and AWS services for CI/CD and hosting:

1. AWS CodePipeline watches the `main` branch through a GitHub connection.
2. AWS CodeBuild runs the API tests, builds the Docker image, and pushes it to Amazon ECR.
3. AWS CloudFormation deploys the image to an Amazon ECS Express Mode service on Fargate.

The ECS service exposes an HTTPS URL. The bundled Iris model is loaded from `models/iris_random_forest_model.joblib`; the service does not need an MLflow server to start. To use MLflow instead, set `MLFLOW_MODEL_URI` and `MLFLOW_TRACKING_URI` on the service.

## Create the pipeline

Use an AWS account and a region that support CodePipeline, CodeBuild, Amazon ECR, and ECS Express Mode. The account needs a default VPC in that region. The AWS identity creating the stack needs permission to create IAM roles, CodePipeline/CodeBuild resources, a GitHub connection, an ECR repository, and ECS resources.

Create a CloudFormation stack from `infra/aws-pipeline.yaml` with these parameters:

- GitHub owner: `SudikshA-0`
- GitHub repository: `MLops_pipeline`
- Branch: `main`
- ECR repository name: `iris-mlops-api`
- Service name: `iris-ml-api`
- Cluster name: `iris-mlops-cluster`
- App stack name: `iris-mlops-app`

When CloudFormation asks, acknowledge that the template creates IAM roles.

The template creates a GitHub connection in `PENDING` state. In the AWS Developer Tools **Connections** page, open that connection and finish the GitHub authorization. Then start `iris-mlops-aws-pipeline` once from the CodePipeline console. After that, pushes to `main` start the AWS pipeline automatically. The previous GitHub Actions workflow is removed, so AWS CodePipeline is the CI/CD pipeline for this repository.

The first successful run creates the ECS Express Mode service. The **ServiceEndpoint** output in the `iris-mlops-app` CloudFormation stack is the API URL. Check `/health` or `/docs` at that URL.

## Cost and cleanup

The running service creates a Fargate task and managed load-balancing resources. CodePipeline, CodeBuild, ECR image storage, CloudWatch logs, and the S3 artifact bucket can also incur charges. Review AWS pricing for the selected region before launching the stack.

To stop the running service, delete the `iris-mlops-app` CloudFormation stack first. Then delete the ECR images and the `iris-mlops-aws-pipeline` stack. The ECR repository and artifact bucket are retained when the pipeline stack is deleted; remove those retained resources from the AWS console if you no longer need them.

## Model training

This deployment pipeline tests and deploys the checked-in model. The existing `train.py` still expects a running MLflow tracking server; automated AWS model training and a hosted MLflow tracking service are separate additions.
