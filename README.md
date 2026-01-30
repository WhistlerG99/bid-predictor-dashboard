# Bid Predictor Dashboard

Interactive Dash application for exploring bid acceptance probabilities from MLflow-logged models and parquet snapshot datasets.

This README is written for an experienced data scientist joining the project. It covers local setup, Docker builds, publishing to AWS ECR, and deploying to AWS App Runner.

## Project overview
- Entry point: `dash_app.py` (Dash server plus callbacks).
- UI helper modules live under `bid_predictor_ui/`.
- The app loads data from local Parquet paths or S3, queries MLflow for a registered model, and optionally uses Redis and Redshift for caching and metadata.
- There are two versions of the dashboard
  - `cache-implementation` -- route-level performance metrics
  - `full-featured-w-cache` -- full featured dashboard w model exploration tools, performance tracking and history    

## Prerequisites
- Python 3.9+ (for local dev).
- Docker (for container builds).
- AWS CLI configured for ECR/App Runner work.
- Access to:
  - Parquet dataset of bid snapshots (local path or S3 prefix).
  - MLflow tracking server and registered model (`MLFLOW_AWS_ARN`).
  - (Optional) Redis for caching.
  - (Optional) Redshift for offer status lookups.
- Optional: a `.env` file in the repository root to store environment variables.

## Local installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Upgrade `pip` and install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Environment configuration
Set the following environment variables directly or in a `.env` file before starting the app.

- `DEFAULT_DATASET_PATH`: default local or mounted parquet path pre-filled in the UI.
- `S3_DATASET_LISTING_URI`: S3 prefix for dataset listing if using S3 (example: `s3://my-bucket/path/to/audit_bid_predictor_csv`).
- `ROLLING_WINDOW_HOURS`: rolling cache window.
- `REDIS_URL`: Redis connection string (example: `redis://localhost:6379` or an ElastiCache endpoint).


**Redshift offer status lookups**
- `REDSHIFT_HOST`
- `REDSHIFT_DATABASE`
- `REDSHIFT_USER`
- `REDSHIFT_PASSWORD`
- `REDSHIFT_PORT` (default: `5439`)

**`full-featured-w-cacahe` Specific environment variables** 
- `MLFLOW_AWS_ARN`: MLflow tracking URI used by the dashboard to load registered models. The app imports this at startup, so it must be set. (not need for `cache-implementation` branch)
- `PERFORMANCE_HISTORY_REFRESH_DAYS`: Refresh offer statuses in performance tracker and history tabs for offers that were active in the last `PERFORMANCE_HISTORY_REFRESH_DAYS` days 
- `PERFORMANCE_HISTORY_S3_URI`: URI for parquet file that stores performance history.

## Running the Dash app locally
From the repository root (with the virtual environment activated), start the dashboard:
```bash
python dash_app.py
```
Then open http://localhost:8000 in your browser. You can adjust dataset paths, MLflow tracking URI, and model details within the UI.

## Docker build and push the Image to AWS
The Dockerfile:
- Uses `python:3.11-slim`.
- Installs Redis tools and starts Redis in the container.
- Runs `python dash_app.py` after launching Redis.

### Building the image

### 1) Define environment variables
```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO_NAME=<repo_name>
export IMAGE_TAG=<version>
export ECR_URI=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

### 2) Create the ECR repository (once)
```bash
aws ecr create-repository \
  --repository-name $REPO_NAME \
  --region $AWS_REGION 2>/dev/null \
|| echo "ECR repo $REPO_NAME already exists"
```

### 3) Authenticate Docker to ECR
```bash
aws ecr get-login-password --region $AWS_REGION \
| docker login --username AWS --password-stdin $ECR_URI
```

### 4) Build and push the image
```bash
docker buildx build \
   --platform linux/amd64 \
   -t $ECR_URI/${REPO_NAME}:${IMAGE_TAG} \
   --push .
```

### (Optional) Sanity Check 
```bash
aws ecr batch-get-image \
  --repository-name ${REPO_NAME} \
  --image-ids imageTag=${IMAGE_TAG} \
  --query 'images[0].imageManifestMediaType' \
  --output text
```
should read something like `application/vnd.docker.distribution.manifest.v2+json` not `application/vnd.oci.*`)


## Deploying to AWS App Runner
### 1) Create a new App Runner service
1. Open **App Runner** in the AWS console and click **Create service**.
2. Source: **Container registry** → **Amazon ECR**.
3. Select the repository and image tag you just pushed.
4. Set the port to `8000`.

### 2) Configure environment variables
In **Runtime environment variables**, add at least:
- `DEFAULT_DATASET_PATH`, `S3_DATASET_LISTING_URI`, `ROLLING_WINDOW_HOURS`, `REDIS_URL`, `REDSHIFT_*`

If you are deploying the `full-featured-w-cache` branch add these as well:
- `MLFLOW_AWS_ARN`, `PERFORMANCE_HISTORY_REFRESH_DAYS`, `PERFORMANCE_HISTORY_S3_URI`


### 3) IAM and networking considerations
- **S3 access**: If your data is in S3, the App Runner service role needs `s3:GetObject` and `s3:ListBucket` on the target bucket/prefix.
- **MLflow access**: If the tracking server is behind a VPC or requires AWS auth, configure an App Runner VPC connector and appropriate IAM role.
- **Redshift access**: If Redshift is in a VPC, App Runner must connect via a VPC connector and security group with access to port 5439.

### 4) Deploy and verify
Deploy the service and monitor startup logs in CloudWatch. Confirm that the MLflow tracking URI resolves and any optional S3/Redis/Redshift integrations are reachable.

## Testing
Run the test suite from the repository root:
```bash
pytest tests
```