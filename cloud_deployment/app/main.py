"""
FastAPI web app for satellite change detection.
Accepts two ZIP uploads (bands per date), runs a chosen method, and
returns a results page with visualizations.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.processor import METHOD_META, run_job

app = FastAPI(title="Satellite Change Detection")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory job store (single instance; use Redis/Firestore for multi-replica)
jobs: dict = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "methods": METHOD_META,
    })


@app.post("/submit")
async def submit(
    background_tasks: BackgroundTasks,
    date1_zip: UploadFile = File(..., description="ZIP containing B02/B03/B04 for date 1"),
    date2_zip: UploadFile = File(..., description="ZIP containing B02/B03/B04 for date 2"),
    method: str = Form(...),
    prompt: Optional[str] = Form(None),
    vlm_mode: str = Form("auto"),
    normalize: str = Form("true"),
):
    if method not in METHOD_META:
        raise HTTPException(400, f"Unknown method '{method}'")

    job_id = uuid.uuid4().hex[:10]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"scd_{job_id}_"))

    d1_path = tmp_dir / "date1.zip"
    d2_path = tmp_dir / "date2.zip"
    d1_path.write_bytes(await date1_zip.read())
    d2_path.write_bytes(await date2_zip.read())

    jobs[job_id] = {
        "status": "running",
        "progress": "Queued…",
        "method": method,
        "method_name": METHOD_META[method]["name"],
    }

    background_tasks.add_task(
        run_job,
        job_id=job_id,
        d1_zip=d1_path,
        d2_zip=d2_path,
        method=method,
        prompt=prompt,
        vlm_mode=vlm_mode,
        normalize=(normalize.lower() == "true"),
        tmp_dir=tmp_dir,
        jobs=jobs,
    )

    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"status": "not_found"}, status_code=404)
    job = jobs[job_id]
    return JSONResponse({
        "status":   job["status"],
        "progress": job.get("progress", ""),
        "error":    job.get("error"),
    })


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results(request: Request, job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete yet")
    return templates.TemplateResponse("result.html", {
        "request": request,
        "job":     job,
        "job_id":  job_id,
    })
