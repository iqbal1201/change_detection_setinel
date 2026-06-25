"""
Method 6 — Vision-Language Model (VLM) Semantic Description.

This module adds a natural-language interpretation layer on top of any binary
change mask produced by Methods 1–5.  It does NOT produce a new raster map;
it answers "what changed and why?" using a frozen VLM.

Supported backends (tried automatically in this order):
  1. Anthropic Claude API  — if ANTHROPIC_API_KEY is set in the environment.
  2. OpenAI GPT-4o API    — if OPENAI_API_KEY is set.
  3. Local LLaVA-1.5-7B  — loaded with 4-bit quantisation via bitsandbytes;
                            requires ~6 GB VRAM (Colab T4 / RTX 3060 sufficient).
  4. Rule-based fallback  — derives a structured description from pixel statistics
                            and spectral pattern analysis.  No GPU or key needed.

Finding-3 caveat (from the literature):
  VLM output is fluent and contextually plausible but not metrologically
  auditable.  Present it as an *interpretation layer*, never as the measured
  result.

Public API
----------
describe(img1, img2, binary, source_method, mode="auto") -> dict
    Returns {"description": str, "backend": str}.
    The description is also written to visualizations/vlm_description.txt.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import label, gaussian_filter

VIZ_DIR = Path(__file__).parents[1] / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

_PROMPT = (
    "You are analyzing a composite satellite image showing a land-surface change "
    "detection result.  The image has three panels side by side:\n"
    "  LEFT  — the scene BEFORE change (T1, multispectral visible bands)\n"
    "  MIDDLE — the scene AFTER change  (T2, same sensor)\n"
    "  RIGHT  — the CHANGE MASK overlaid in red on T2 (red = detected change)\n\n"
    "Please provide a structured analysis with these five sections:\n"
    "1. CHANGE TYPE: What kind of surface change is visible? "
    "(e.g. vegetation clearance, construction, flooding, mining, burn scar)\n"
    "2. SPATIAL EXTENT: How much area is affected? Concentrated or dispersed?\n"
    "3. CHANGE PATTERN: Shape and arrangement of changed areas.\n"
    "4. LIKELY CAUSE: Most plausible land-use activity or event.\n"
    "5. CONFIDENCE: Your confidence level and the specific visual evidence supporting it.\n\n"
    "Important: this is zero-shot inference from visible-band imagery only. "
    "State uncertainty explicitly — do not hallucinate NIR or SWIR information."
)


# ── Panel image builder ───────────────────────────────────────────────────────

def _to_uint8_rgb(img: np.ndarray, percentile: int = 2) -> np.ndarray:
    """Convert (H, W, 3) float32 [B, G, R] stack to uint8 RGB."""
    rgb = img[:, :, [2, 1, 0]].copy().astype(np.float32)
    np.nan_to_num(rgb, nan=0.0, copy=False)
    lo = np.percentile(rgb, percentile)
    hi = np.percentile(rgb, 100 - percentile)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _make_panel_image(
    img1: np.ndarray,
    img2: np.ndarray,
    binary: np.ndarray,
    panel_width: int = 512,
) -> Image.Image:
    """
    Build a three-panel RGB image: [Before | After | Change overlay].
    Saved as a PIL Image so it can be base64-encoded or passed to a local model.
    """
    # Downsample large Sentinel-2 tiles before building the panel to avoid OOM
    h, w = img1.shape[:2]
    step = max(1, max(h, w) // 1024)
    if step > 1:
        img1   = img1  [::step, ::step]
        img2   = img2  [::step, ::step]
        binary = binary[::step, ::step]

    rgb1 = _to_uint8_rgb(img1)
    rgb2 = _to_uint8_rgb(img2)

    # Change overlay: T2 with red mask
    overlay = rgb2.copy()
    overlay[binary == 1, 0] = 220   # R channel → red
    overlay[binary == 1, 1] = (overlay[binary == 1, 1] * 0.3).astype(np.uint8)
    overlay[binary == 1, 2] = (overlay[binary == 1, 2] * 0.3).astype(np.uint8)

    h = rgb1.shape[0]
    target_h = panel_width  # square panels
    scale = target_h / h
    new_w = int(rgb1.shape[1] * scale)

    def _resize(arr):
        pil = Image.fromarray(arr)
        return pil.resize((new_w, target_h), Image.LANCZOS)

    panels = [_resize(rgb1), _resize(rgb2), _resize(overlay)]
    combined = Image.new("RGB", (new_w * 3, target_h))
    for i, p in enumerate(panels):
        combined.paste(p, (i * new_w, 0))

    return combined


def _encode_image(pil_img: Image.Image, quality: int = 85) -> str:
    """Return base64-encoded JPEG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── Backend 1 — Anthropic Claude API ─────────────────────────────────────────

def _describe_anthropic(panel_image: Image.Image) -> str:
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    img_b64 = _encode_image(panel_image)

    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": _PROMPT},
            ],
        }],
    )
    return msg.content[0].text


# ── Backend 2 — OpenAI GPT-4o API ────────────────────────────────────────────

def _describe_openai(panel_image: Image.Image) -> str:
    import openai  # pip install openai

    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=20.0,   # fail fast if network is blocked — avoids long hang
    )
    img_b64 = _encode_image(panel_image)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
                {"type": "text", "text": _PROMPT},
            ],
        }],
    )
    return resp.choices[0].message.content


# ── Backend 3 — Local LLaVA-1.5-7B (4-bit) ──────────────────────────────────

def _describe_llava(panel_image: Image.Image) -> str:
    """
    Load LLaVA-1.5-7B with 4-bit quantisation (bitsandbytes) and run inference.
    Requires: pip install transformers bitsandbytes accelerate
    Needs ~6 GB VRAM.
    """
    import torch
    from transformers import (
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration,
        BitsAndBytesConfig,
    )

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    print(f"  [VLM] Loading {model_id} with 4-bit quantisation...")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print("  [VLM] Model loaded.")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(images=panel_image, text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
        )

    generated = processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return generated.strip()


# ── Backend 4 — Rule-based fallback ──────────────────────────────────────────

def _describe_rule_based(
    img1: np.ndarray,
    img2: np.ndarray,
    binary: np.ndarray,
    source_method: str,
) -> str:
    """
    Derive a structured description from pixel statistics and spectral analysis.
    No GPU or API key required.
    """
    h, w = binary.shape
    total_px   = h * w
    changed_px = int(binary.sum())
    change_pct = 100.0 * changed_px / total_px

    # ── Spatial clustering ──────────────────────────────────────────────────
    labeled, n_clusters = label(binary)
    if n_clusters > 0 and changed_px > 0:
        sizes = np.bincount(labeled.ravel())[1:]  # skip background (label 0)
        largest_pct = 100.0 * sizes.max() / changed_px
        compactness = "compact" if largest_pct > 60 else "fragmented"
    else:
        n_clusters    = 0
        largest_pct   = 0.0
        compactness   = "absent"

    # ── Centroid location ────────────────────────────────────────────────────
    if changed_px > 0:
        ys, xs  = np.where(binary == 1)
        rel_y   = ys.mean() / h
        rel_x   = xs.mean() / w
        v_label = "northern" if rel_y < 0.4 else ("southern" if rel_y > 0.6 else "central")
        h_label = "western"  if rel_x < 0.4 else ("eastern"  if rel_x > 0.6 else "")
        location_str = f"{v_label} {h_label}".strip() + " portion of the scene"
    else:
        location_str = "no clear spatial cluster"

    # ── Spectral change analysis (BGR bands) ─────────────────────────────────
    if changed_px > 0:
        px1 = img1[binary == 1].astype(np.float32)   # (N, 3) BGR
        px2 = img2[binary == 1].astype(np.float32)
        d   = px2 - px1
        db, dg, dr = d[:, 0].mean(), d[:, 1].mean(), d[:, 2].mean()

        # Spectral signature → change type hint
        if dr > 0.03 and dg < -0.02:
            spectral_hint = (
                "increased red reflectance with decreased green — consistent with "
                "vegetation loss, bare-soil exposure, or construction activity"
            )
        elif dr < -0.02 and dg < -0.02 and db < -0.02:
            spectral_hint = (
                "reflectance decreased across all visible bands — consistent with "
                "water-body expansion, flooding, or deep-shadow increase"
            )
        elif dr > 0.02 and dg > 0.02 and db > 0.02:
            spectral_hint = (
                "reflectance increased across all visible bands — consistent with "
                "surface clearing, new impervious cover, or dry-season soil exposure"
            )
        elif dg > 0.02 and dr < 0:
            spectral_hint = (
                "increased green with decreased red — consistent with "
                "vegetation regrowth or crop establishment"
            )
        else:
            spectral_hint = (
                "mixed or subtle spectral shifts; change type is ambiguous "
                "from visible bands alone"
            )

        spectral_line = (
            f"Spectral change in changed pixels: "
            f"ΔBlue={db:+.3f}, ΔGreen={dg:+.3f}, ΔRed={dr:+.3f}.\n"
            f"Spectral interpretation: {spectral_hint}."
        )
    else:
        spectral_line = "No changed pixels detected — spectral analysis not applicable."

    description = f"""Rule-based spectral analysis  (source: {source_method})

1. CHANGE TYPE:
   {spectral_hint if changed_px > 0 else "No change detected."}

2. SPATIAL EXTENT:
   {change_pct:.1f}% of the scene ({changed_px:,} / {total_px:,} pixels) is flagged as changed.

3. CHANGE PATTERN:
   {n_clusters} spatially distinct changed region(s) detected.
   Pattern is {compactness} — the largest cluster accounts for {largest_pct:.0f}% of all
   changed area.  Changes are concentrated in the {location_str}.

4. LIKELY CAUSE:
   Based on visible-band spectral signatures only:
   {spectral_hint.capitalize() if changed_px > 0 else "Cannot determine cause — no change detected."}
   A definitive attribution requires NIR/SWIR bands or contextual ground truth.

5. CONFIDENCE:
   LOW — this description is derived purely from pixel statistics and simple
   spectral heuristics, without a trained language or vision model.  It should
   be treated as a first-pass screening aid, not an authoritative interpretation.
   {spectral_line}
"""
    return description.strip()


# ── Auto-dispatch ─────────────────────────────────────────────────────────────

def describe(
    img1: np.ndarray,
    img2: np.ndarray,
    binary: np.ndarray,
    source_method: str = "unknown",
    mode: str = "auto",
) -> Dict[str, str]:
    """
    Generate a natural-language semantic description of the detected change.

    Parameters
    ----------
    img1, img2    : (H, W, 3) float32 [B02, B03, B04]
    binary        : (H, W) uint8 {0, 1}  change mask from any method
    source_method : name of the method that produced `binary` (for attribution)
    mode          : "auto" | "anthropic" | "openai" | "llava" | "rule_based"
                    "auto" tries each backend in order and uses the first that works.

    Returns
    -------
    dict with keys:
      "description" : str   — the natural-language output
      "backend"     : str   — which backend was used
    """
    print(f"  [VLM] Building panel image for semantic description...")
    panel = _make_panel_image(img1, img2, binary)

    backends = []
    if mode == "auto":
        _oai_key = os.environ.get("OPENAI_API_KEY", "")
        if _oai_key and _oai_key != "placeholder":
            backends.append(("openai",   lambda: _describe_openai(panel)))
        backends.append(("rule_based", lambda: _describe_rule_based(img1, img2, binary, source_method)))
    elif mode == "anthropic":
        backends = [("anthropic", lambda: _describe_anthropic(panel))]
    elif mode == "openai":
        backends = [("openai",    lambda: _describe_openai(panel))]
    elif mode == "llava":
        backends = [("llava",     lambda: _describe_llava(panel))]
    else:  # rule_based
        backends = [("rule_based", lambda: _describe_rule_based(img1, img2, binary, source_method))]

    description = ""
    used_backend = "none"

    for name, fn in backends:
        try:
            print(f"  [VLM] Trying backend: {name}...")
            description = fn()
            used_backend = name
            print(f"  [VLM] Success with backend: {name}")
            break
        except Exception as e:
            print(f"  [VLM] Backend '{name}' failed: {e}")
            continue

    if not description:
        description = _describe_rule_based(img1, img2, binary, source_method)
        used_backend = "rule_based_fallback"

    # Save panel image
    panel_path = VIZ_DIR / "vlm_panel.jpg"
    panel.save(str(panel_path), quality=90)
    print(f"  [VLM] Panel image saved → {panel_path.name}")

    # Save description text
    out_path = VIZ_DIR / "vlm_description.txt"
    out_path.write_text(
        f"Backend: {used_backend}\n"
        f"Source method: {source_method}\n"
        f"{'='*60}\n\n"
        + description,
        encoding="utf-8",
    )
    print(f"  [VLM] Description saved → {out_path.name}")

    return {"description": description, "backend": used_backend}
