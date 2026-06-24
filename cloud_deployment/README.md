# Cloud Deployment — Satellite Change Detection

Deploys the change detection pipeline as a web application on **Google Cloud Run**, with images stored in **Artifact Registry** and builds automated by **Cloud Build**.

---

## Architecture

```
Developer pushes to main
         │
         ▼
  ┌─────────────────────────────────────────────────┐
  │  Cloud Build  (cloudbuild.yaml)                 │
  │                                                 │
  │  1. docker build  (Dockerfile)                  │
  │     – copies methods/, utils/, app/ into image  │
  │                                                 │
  │  2. docker push                                 │
  │     → Artifact Registry (scd-images repo)       │
  │                                                 │
  │  3. gcloud run deploy                           │
  │     → Cloud Run (scd-app service)               │
  └─────────────────────────────────────────────────┘
         │
         ▼
  Cloud Run serves FastAPI web app
  https://scd-app-xxxx.run.app
```

---

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- A GCP project with a billing account attached
- Owner or Editor role on the project (for initial setup)

```bash
gcloud auth login
gcloud auth application-default login
```

---

## Folder Structure

```
cloud_deployment/
├── README.md          ← this file
├── setup.sh           ← run once to create GCP resources
├── Dockerfile         ← container image definition
├── requirements.txt   ← Python dependencies
├── service.yaml       ← Cloud Run service spec (reference)
└── app/
    ├── main.py        ← FastAPI routes
    ├── processor.py   ← background job runner (imports methods/ and utils/)
    ├── templates/
    │   ├── index.html ← upload form + method selector
    │   └── result.html← results page (figure, map, VLM text)
    └── static/
        ├── style.css
        └── app.js

# At project root (not inside cloud_deployment/):
cloudbuild.yaml        ← CI/CD pipeline definition
methods/               ← all detection algorithms (unchanged)
utils/                 ← shared utilities (unchanged)
```

> The web app does **not** duplicate any algorithm code. `processor.py` imports directly
> from `methods/` and `utils/`, which are copied into the container by the Dockerfile.

---

## Step 1 — One-time GCP Setup

Run `setup.sh` once to provision all required GCP resources.

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=asia-southeast1   # change to your preferred region

bash cloud_deployment/setup.sh
```

### What setup.sh does

| # | Action |
|---|--------|
| 1 | Enables APIs: Cloud Run, Artifact Registry, Cloud Build, Secret Manager, IAM |
| 2 | Creates Artifact Registry Docker repository (`scd-images`) |
| 3 | Creates a dedicated Cloud Run runtime service account (`scd-run-sa`) |
| 4 | Grants Cloud Build SA the roles it needs (see IAM section below) |
| 5 | Grants runtime SA access to Secret Manager |
| 6 | Creates `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` secrets (prompts for values) |
| 7 | Prints the exact trigger configuration to use in step 2 |

---

## Step 2 — Connect Repository to Cloud Build

1. Open [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers) in GCP Console
2. Click **Connect Repository** and link your Git repo (GitHub / GitLab / Cloud Source Repos)
3. Click **Create Trigger** with these settings:

| Setting | Value |
|---------|-------|
| Event | Push to branch |
| Branch | `^main$` |
| Configuration | Cloud Build configuration file |
| Location | Repository → `cloudbuild.yaml` |

4. Under **Substitution variables**, add:

| Variable | Value |
|----------|-------|
| `_REGION` | `asia-southeast1` |
| `_REPO` | `scd-images` |
| `_SERVICE` | `scd-app` |
| `_RUN_SA` | `scd-run-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com` |

---

## Step 3 — First Deployment

Either push a commit to `main`, or trigger the build manually:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_REGION=asia-southeast1,_REPO=scd-images,_SERVICE=scd-app,_RUN_SA=scd-run-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  .
```

> Run this from the **project root**, not from inside `cloud_deployment/`.
> The build context must be the root so Docker can `COPY methods/` and `COPY utils/`.

### What Cloud Build does (cloudbuild.yaml)

```
Step 1 — build   docker build -f cloud_deployment/Dockerfile .
Step 2 — push    docker push → Artifact Registry (tagged :COMMIT_SHA and :latest)
Step 3 — deploy  gcloud run deploy scd-app --image ...:COMMIT_SHA
```

---

## Step 4 — Get the Service URL

```bash
gcloud run services describe scd-app \
  --region asia-southeast1 \
  --format='value(status.url)'
```

Open the URL in a browser. You should see the upload interface.

---

## IAM — Service Account Roles

Two service accounts are used:

### Cloud Build SA — `{PROJECT_NUMBER}@cloudbuild.gserviceaccount.com`

| Role | Required for |
|------|-------------|
| `roles/artifactregistry.writer` | Push Docker image (Step 2) |
| `roles/run.admin` | Deploy Cloud Run service (Step 3) |
| `roles/iam.serviceAccountUser` | Assign runtime SA to the service (Step 3) |

### Cloud Run Runtime SA — `scd-run-sa@PROJECT_ID.iam.gserviceaccount.com`

| Role | Required for |
|------|-------------|
| `roles/secretmanager.secretAccessor` | Read `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` at startup |

All bindings are applied automatically by `setup.sh`.

---

## Secrets

API keys for Method 6 (VLM) are stored in **Secret Manager** and injected as environment variables at container startup. They are never baked into the image.

| Secret name | Used by |
|-------------|---------|
| `OPENAI_API_KEY` | Method 6 — GPT-4o vision backend |

To update the secret after initial setup:

```bash
echo -n "sk-..." | gcloud secrets versions add OPENAI_API_KEY --data-file=-
```

If you do not have API keys, Method 6 will fall back to the local `rule_based` backend automatically.

---

## Cloud Run Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Memory | 4 Gi | rasterio + sklearn + matplotlib peak usage |
| CPU | 2 | background thread processing |
| Request timeout | 900 s | deep methods (DINOv2, SAM2) can take several minutes |
| Concurrency | 1 | one background job per instance (in-memory job store) |
| Min instances | 0 | scale to zero when idle |
| Max instances | 5 | horizontal scale for concurrent users |

---

## Subsequent Deployments

After the trigger is set up, every push to `main` automatically:

1. Builds a new image tagged with the commit SHA
2. Pushes it to Artifact Registry
3. Deploys the new revision to Cloud Run with zero downtime

---

## Local Development

To run the web app locally without Docker:

```bash
# From the project root
pip install -r cloud_deployment/requirements.txt

cd "path/to/solafune_change detection"
uvicorn app.main:app --reload --port 8080
```

Then open `http://localhost:8080`.

To run in Docker locally:

```bash
# From the project root
docker build -f cloud_deployment/Dockerfile -t scd-app .
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OPENAI_API_KEY=sk-... \
  scd-app
```
