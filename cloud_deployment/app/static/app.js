/* ── File upload: show filename and highlight zone ─────────────── */
function wireUpload(inputId, zoneId, labelId) {
  const input = document.getElementById(inputId);
  const zone  = document.getElementById(zoneId);
  const label = document.getElementById(labelId);

  if (!input || !zone || !label) return;

  input.addEventListener("change", () => {
    if (input.files.length) {
      label.textContent = "✓ " + input.files[0].name;
      zone.classList.add("has-file");
    }
  });

  // Drag-and-drop
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".zip")) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      label.textContent = "✓ " + file.name;
      zone.classList.add("has-file");
    }
  });
}

wireUpload("date1_zip", "zone1", "fname1");
wireUpload("date2_zip", "zone2", "fname2");

/* ── VLM prompt block visibility ──────────────────────────────── */
const methodRadios = document.querySelectorAll('input[name="method"]');
const vlmBlock     = document.getElementById("vlmBlock");

function toggleVlm() {
  const selected = document.querySelector('input[name="method"]:checked');
  if (selected && selected.value === "6") {
    vlmBlock.classList.remove("hidden");
  } else {
    vlmBlock.classList.add("hidden");
  }
}

methodRadios.forEach((r) => r.addEventListener("change", toggleVlm));
toggleVlm(); // run on load

/* ── Form submission ──────────────────────────────────────────── */
const form       = document.getElementById("submitForm");
const overlay    = document.getElementById("overlay");
const overlayTitle  = document.getElementById("overlayTitle");
const overlayStatus = document.getElementById("overlayStatus");
const progressFill  = document.getElementById("progressFill");
const submitBtn     = document.getElementById("submitBtn");
const btnText       = document.getElementById("btnText");
const btnSpinner    = document.getElementById("btnSpinner");

const PROGRESS_LABELS = [
  { pct: 10,  msg: "Extracting images…" },
  { pct: 30,  msg: "Normalizing radiometry…" },
  { pct: 50,  msg: "Running detection method…" },
  { pct: 75,  msg: "Vectorizing results…" },
  { pct: 90,  msg: "Building visualizations…" },
];

function setOverlayProgress(pct) {
  progressFill.style.width = pct + "%";
}

function showOverlay(methodName) {
  overlayTitle.textContent  = "Running " + methodName + "…";
  overlayStatus.textContent = "Starting…";
  setOverlayProgress(5);
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

let pollTimer = null;

async function pollStatus(jobId) {
  let stepIdx = 0;

  pollTimer = setInterval(async () => {
    try {
      const res  = await fetch("/status/" + jobId);
      const data = await res.json();

      overlayStatus.textContent = data.progress || "Processing…";

      // Advance fake progress bar
      if (stepIdx < PROGRESS_LABELS.length) {
        setOverlayProgress(PROGRESS_LABELS[stepIdx].pct);
        stepIdx++;
      }

      if (data.status === "done") {
        clearInterval(pollTimer);
        setOverlayProgress(100);
        overlayStatus.textContent = "Complete! Redirecting…";
        setTimeout(() => {
          window.location.href = "/results/" + jobId;
        }, 600);
      } else if (data.status === "error") {
        clearInterval(pollTimer);
        hideOverlay();
        resetSubmitBtn();
        alert("Processing failed:\n\n" + (data.error || "Unknown error"));
      }
    } catch (err) {
      // Network hiccup — keep polling
    }
  }, 1500);
}

function resetSubmitBtn() {
  submitBtn.disabled = false;
  btnText.textContent = "Run Analysis";
  btnSpinner.classList.add("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const d1 = document.getElementById("date1_zip").files[0];
  const d2 = document.getElementById("date2_zip").files[0];

  if (!d1 || !d2) {
    alert("Please upload ZIP files for both dates.");
    return;
  }

  const methodEl = document.querySelector('input[name="method"]:checked');
  if (!methodEl) {
    alert("Please select a detection method.");
    return;
  }

  const methodName = methodEl.closest(".method-card")
    .querySelector(".method-name").textContent;

  // Disable button
  submitBtn.disabled = true;
  btnText.textContent = "Submitting…";
  btnSpinner.classList.remove("hidden");
  showOverlay(methodName);

  const fd = new FormData(form);
  // Ensure normalize value is sent correctly
  if (!document.getElementById("normalizeCheck").checked) {
    fd.set("normalize", "false");
  } else {
    fd.set("normalize", "true");
  }

  try {
    const res  = await fetch("/submit", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }
    const data = await res.json();
    btnText.textContent = "Processing…";
    pollStatus(data.job_id);
  } catch (err) {
    hideOverlay();
    resetSubmitBtn();
    alert("Submission failed:\n\n" + err.message);
  }
});
