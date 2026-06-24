#!/usr/bin/env bash
# ── One-time GCP infrastructure setup ────────────────────────────────────────
# Run this once before the first Cloud Build trigger fires.
# Prerequisites: gcloud CLI installed and authenticated (gcloud auth login)
#
# Usage:
#   export PROJECT_ID=your-gcp-project-id
#   export REGION=asia-southeast1
#   bash cloud_deployment/setup.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID env var}"
REGION="${REGION:-asia-southeast1}"
REPO="scd-images"
SERVICE="scd-app"
RUN_SA_NAME="scd-run-sa"
RUN_SA="${RUN_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Project : $PROJECT_ID"
echo "==> Region  : $REGION"
echo "==> Repo    : $REPO"
echo "==> Service : $SERVICE"
echo "==> Run SA  : $RUN_SA"
echo ""

gcloud config set project "$PROJECT_ID"

# ── 1. Enable required APIs ───────────────────────────────────────────────────
echo "==> Enabling GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com

# ── 2. Create Artifact Registry repository ────────────────────────────────────
echo "==> Creating Artifact Registry repo: $REPO..."
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Satellite Change Detection app images" \
  || echo "    (repo may already exist — skipping)"

# ── 3. Create Cloud Run runtime service account ───────────────────────────────
echo "==> Creating runtime service account: $RUN_SA..."
gcloud iam service-accounts create "$RUN_SA_NAME" \
  --display-name="Satellite Change Detection — Cloud Run runtime" \
  || echo "    (SA may already exist — skipping)"

# ── 4. Grant Cloud Build SA permissions ───────────────────────────────────────
echo "==> Granting Cloud Build SA permissions..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

# Cloud Build needs to push images and deploy to Cloud Run
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/run.admin"

# Cloud Build must be able to act as the Cloud Run runtime SA
gcloud iam service-accounts add-iam-policy-binding "$RUN_SA" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/iam.serviceAccountUser"

# ── 5. Grant runtime SA permission to read secrets ────────────────────────────
echo "==> Granting runtime SA secret access..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/secretmanager.secretAccessor"

# ── 6. Create API key secrets ─────────────────────────────────────────────────
echo "==> Creating Secret Manager secrets (enter API keys or press Enter to skip)..."

create_secret() {
  local name="$1"
  echo -n "    ${name} value (leave blank to skip): "
  read -rs value
  echo ""
  if [ -n "$value" ]; then
    echo -n "$value" | gcloud secrets create "$name" \
      --replication-policy=automatic \
      --data-file=- \
      2>/dev/null \
    || echo -n "$value" | gcloud secrets versions add "$name" --data-file=-
    echo "    ✓ Secret '$name' stored."
  else
    # Create an empty placeholder so Cloud Run deploy doesn't fail
    echo -n "placeholder" | gcloud secrets create "$name" \
      --replication-policy=automatic \
      --data-file=- \
      2>/dev/null || true
    echo "    ! '$name' set to placeholder — update it before using Method 6."
  fi
}

create_secret "OPENAI_API_KEY"

# ── 7. Connect Cloud Build to repository ──────────────────────────────────────
echo ""
echo "==> Next steps:"
echo ""
echo "  1. Connect your Git repo in Cloud Build:"
echo "     https://console.cloud.google.com/cloud-build/triggers?project=${PROJECT_ID}"
echo ""
echo "  2. Create a trigger with these settings:"
echo "       Event        : Push to branch (^main$)"
echo "       Config file  : cloudbuild.yaml  (repo root)"
echo "       Substitutions:"
echo "         _REGION  = ${REGION}"
echo "         _REPO    = ${REPO}"
echo "         _SERVICE = ${SERVICE}"
echo "         _RUN_SA  = ${RUN_SA}"
echo ""
echo "  3. First manual build (or push a commit to main):"
echo "     gcloud builds submit --config cloudbuild.yaml \\"
echo "       --substitutions=_REGION=${REGION},_REPO=${REPO},_SERVICE=${SERVICE},_RUN_SA=${RUN_SA} ."
echo ""
echo "  4. After deploy, find your service URL:"
echo "     gcloud run services describe ${SERVICE} --region ${REGION} --format='value(status.url)'"
echo ""
echo "==> Setup complete."
