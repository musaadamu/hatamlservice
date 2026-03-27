# ============================================
# HATA ML Service - AWS Lambda Deployment Script
# Run from within the ml-service directory
# ============================================
#
# PREREQUISITES:
#   1. AWS CLI installed and configured (aws configure)
#   2. Docker Desktop installed and running
#   3. Your AWS Account ID (find in AWS Console top-right)
#
# USAGE:
#   cd ml-service
#   # Edit this script first — replace YOUR_ACCOUNT_ID with your actual AWS account ID
#   powershell -ExecutionPolicy Bypass -File deploy-lambda.ps1
# ============================================

# ---- CONFIGURATION (EDIT THESE) ----
$AWS_ACCOUNT_ID = "YOUR_ACCOUNT_ID"  # ← Replace with your 12-digit AWS Account ID
$AWS_REGION = "us-east-1"
$REPO_NAME = "hata-mlservice"
$FUNCTION_NAME = "hata-mlservice"
$LAMBDA_MEMORY = 3008  # MB — sufficient for quantized model + PyTorch + LIME
$LAMBDA_TIMEOUT = 300  # seconds (5 minutes max for LIME processing)
$HF_TOKEN = "YOUR_HF_TOKEN"  # ← Replace with your HuggingFace token (for downloading model)

# Derived values
$ECR_URI = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
$IMAGE_URI = "$ECR_URI/${REPO_NAME}:latest"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HATA ML Service - Lambda Deployment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ---- STEP 1: Validate configuration ----
if ($AWS_ACCOUNT_ID -eq "YOUR_ACCOUNT_ID") {
    Write-Host "ERROR: Please edit this script and replace YOUR_ACCOUNT_ID with your actual AWS Account ID" -ForegroundColor Red
    Write-Host "Find your Account ID in the AWS Console (top-right corner)" -ForegroundColor Yellow
    exit 1
}

if ($HF_TOKEN -eq "YOUR_HF_TOKEN") {
    Write-Host "WARNING: HF_TOKEN is not set. The model will not be able to download from HuggingFace Hub." -ForegroundColor Yellow
    Write-Host "Set it now or update it later in the Lambda console." -ForegroundColor Yellow
}

# ---- STEP 2: Create ECR Repository (skip if exists) ----
Write-Host "`n[Step 1/7] Creating ECR repository..." -ForegroundColor Green
try {
    aws ecr describe-repositories --repository-names $REPO_NAME --region $AWS_REGION 2>$null
    Write-Host "  ECR repository '$REPO_NAME' already exists." -ForegroundColor Yellow
} catch {
    aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION
    Write-Host "  ECR repository '$REPO_NAME' created." -ForegroundColor Green
}

# ---- STEP 3: Authenticate Docker to ECR ----
Write-Host "`n[Step 2/7] Authenticating Docker to ECR..." -ForegroundColor Green
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker authentication failed. Is Docker Desktop running?" -ForegroundColor Red
    exit 1
}
Write-Host "  Docker authenticated to ECR." -ForegroundColor Green

# ---- STEP 4: Build Docker image ----
Write-Host "`n[Step 3/7] Building Docker image..." -ForegroundColor Green
docker build -t $REPO_NAME .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Docker image built successfully." -ForegroundColor Green

# ---- STEP 5: Tag and push to ECR ----
Write-Host "`n[Step 4/7] Pushing image to ECR..." -ForegroundColor Green
docker tag "${REPO_NAME}:latest" $IMAGE_URI
docker push $IMAGE_URI
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker push failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Image pushed to ECR: $IMAGE_URI" -ForegroundColor Green

# ---- STEP 6: Create or Update Lambda function ----
Write-Host "`n[Step 5/7] Creating/Updating Lambda function..." -ForegroundColor Green
$existingFunction = $null
try {
    $existingFunction = aws lambda get-function --function-name $FUNCTION_NAME --region $AWS_REGION 2>$null
} catch {}

if ($existingFunction) {
    Write-Host "  Lambda function exists — updating code..." -ForegroundColor Yellow
    aws lambda update-function-code `
        --function-name $FUNCTION_NAME `
        --image-uri $IMAGE_URI `
        --region $AWS_REGION

    # Wait for update to complete
    Write-Host "  Waiting for function update to complete..." -ForegroundColor Yellow
    aws lambda wait function-updated --function-name $FUNCTION_NAME --region $AWS_REGION

    # Update configuration
    aws lambda update-function-configuration `
        --function-name $FUNCTION_NAME `
        --memory-size $LAMBDA_MEMORY `
        --timeout $LAMBDA_TIMEOUT `
        --environment "Variables={HF_TOKEN=$HF_TOKEN,USE_HF_INFERENCE_API=true,LOG_LEVEL=INFO}" `
        --region $AWS_REGION
} else {
    Write-Host "  Creating new Lambda function..." -ForegroundColor Green
    
    # Check if execution role exists
    $ROLE_ARN = "arn:aws:iam::${AWS_ACCOUNT_ID}:role/lambda-execution-role"
    Write-Host "  Using IAM role: $ROLE_ARN" -ForegroundColor Yellow
    Write-Host "  NOTE: If this role doesn't exist, create it first (see deployment guide)" -ForegroundColor Yellow

    aws lambda create-function `
        --function-name $FUNCTION_NAME `
        --package-type Image `
        --code "ImageUri=$IMAGE_URI" `
        --role $ROLE_ARN `
        --memory-size $LAMBDA_MEMORY `
        --timeout $LAMBDA_TIMEOUT `
        --environment "Variables={HF_TOKEN=$HF_TOKEN,USE_HF_INFERENCE_API=true,LOG_LEVEL=INFO}" `
        --region $AWS_REGION
}
Write-Host "  Lambda function ready." -ForegroundColor Green

# ---- STEP 7: Set up public Function URL ----
Write-Host "`n[Step 6/7] Setting up public Function URL..." -ForegroundColor Green
try {
    $urlConfig = aws lambda get-function-url-config --function-name $FUNCTION_NAME --region $AWS_REGION 2>$null | ConvertFrom-Json
    Write-Host "  Function URL already exists: $($urlConfig.FunctionUrl)" -ForegroundColor Yellow
} catch {
    # Add permission for public access
    aws lambda add-permission `
        --function-name $FUNCTION_NAME `
        --statement-id FunctionURLAllowPublicAccess `
        --action lambda:InvokeFunctionUrl `
        --principal "*" `
        --function-url-auth-type NONE `
        --region $AWS_REGION 2>$null

    # Create function URL
    $urlConfig = aws lambda create-function-url-config `
        --function-name $FUNCTION_NAME `
        --auth-type NONE `
        --cors "AllowOrigins=*,AllowMethods=*,AllowHeaders=*" `
        --region $AWS_REGION | ConvertFrom-Json

    Write-Host "  Function URL created: $($urlConfig.FunctionUrl)" -ForegroundColor Green
}

# ---- STEP 8: Print results ----
$functionUrl = $urlConfig.FunctionUrl
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Lambda Function URL:" -ForegroundColor White
Write-Host "  $functionUrl" -ForegroundColor Yellow
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor White
Write-Host "  1. Test the service:" -ForegroundColor White
Write-Host "     curl $functionUrl" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Update your backend .env file:" -ForegroundColor White
Write-Host "     ML_SERVICE_URL=$functionUrl" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Redeploy your Express backend on Render" -ForegroundColor White
Write-Host ""
Write-Host "  4. (Optional) Set up CloudWatch keep-warm rule:" -ForegroundColor White
Write-Host "     Every 5 min, call: ${functionUrl}ping" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan
