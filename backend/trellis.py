"""TRELLIS 2D-to-3D generation via HuggingFace gradio_client.

Connects to the ``trellis-community/TRELLIS`` ZeroGPU Space on HuggingFace
and runs the full generation pipeline:
  1. Resize + save image
  2. Start session on the Space
  3. Preprocess (background removal)
  4. Initialise tab state
  5. Resolve seed
  6. Generate GLB (async job, poll until done)
  7. Copy result to output_dir and return path

Usage from FastAPI::

    from reform3d.trellis import generate_glb
    glb_path = await asyncio.to_thread(generate_glb, image_bytes, hf_token, output_dir)
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — fail gracefully so the rest of the app still boots
# ---------------------------------------------------------------------------
try:
    from gradio_client import Client, handle_file  # type: ignore

    HAS_GRADIO_CLIENT = True
except ImportError:
    HAS_GRADIO_CLIENT = False
    logger.warning(
        "gradio-client not installed — TRELLIS endpoint will be unavailable. "
        "Run: pip install gradio-client"
    )

try:
    import httpx  # noqa: F401 — patch timeout for long ZeroGPU tasks

    original_httpx_init = httpx.Client.__init__

    def _patched_httpx_init(self, *args, **kwargs):  # type: ignore
        kwargs["timeout"] = httpx.Timeout(600.0)  # 10-minute hard timeout
        original_httpx_init(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_httpx_init  # type: ignore
except ImportError:
    pass  # httpx not required at module level


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_client(hf_token: str) -> "Client":
    """Create a Gradio client for TRELLIS with 3 retry attempts."""
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    for attempt in range(3):
        try:
            return Client("trellis-community/TRELLIS")
        except Exception as exc:
            logger.warning(
                "[Trellis] Client init attempt %d/3 failed: %s", attempt + 1, exc
            )
            if attempt == 2:
                raise
            time.sleep(3)

    raise RuntimeError("Failed to connect to TRELLIS after 3 attempts")  # unreachable


def _resize_and_save(image_bytes: bytes, dest_path: str) -> None:
    """Save image bytes to *dest_path*, resizing to ≤1024px on the longest side."""
    try:
        from PIL import Image as PILImage  # type: ignore

        import io

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")
        max_side = 1024
        if img.width > max_side or img.height > max_side:
            img.thumbnail((max_side, max_side), PILImage.LANCZOS)
            logger.info("[Trellis] Resized image to %s", img.size)
        img.save(dest_path, "PNG")
    except ImportError:
        # Pillow unavailable — write raw bytes
        with open(dest_path, "wb") as fh:
            fh.write(image_bytes)
    except Exception as exc:
        logger.warning("[Trellis] Resize failed (%s), using raw bytes.", exc)
        with open(dest_path, "wb") as fh:
            fh.write(image_bytes)


def _to_data_uri(path: str) -> str:
    """Encode a PNG file as a base64 data URI (required by newer Gradio API)."""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_glb_path(result) -> Optional[str]:
    """Recursively pull the first .glb path out of the generate_and_extract_glb result."""

    def _from_item(item) -> Optional[str]:
        if isinstance(item, str) and item.lower().endswith(".glb"):
            return item
        if isinstance(item, dict):
            for key in ("path", "url"):
                val = item.get(key) or ""
                if val.lower().endswith(".glb"):
                    return val
        return None

    if isinstance(result, (list, tuple)):
        # Try from the end first — download button is last and most reliable
        for item in reversed(result):
            found = _from_item(item)
            if found:
                return found
        # Fallback: scan forwards
        for item in result:
            found = _from_item(item)
            if found:
                return found
    else:
        return _from_item(result)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_glb(
    image_bytes: bytes,
    hf_token: str,
    output_dir: Path,
) -> Path:
    """Run the full TRELLIS pipeline and return the path to the saved .glb file.

    This is a **blocking** function and should be called via ``asyncio.to_thread``
    from an async FastAPI endpoint to avoid blocking the event loop.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        hf_token: HuggingFace access token for ZeroGPU authentication.
        output_dir: Directory where the output .glb will be saved.

    Returns:
        Path to the generated and saved ``.glb`` file.

    Raises:
        RuntimeError: If ``gradio-client`` is not installed.
        RuntimeError: If TRELLIS returns no GLB file.
    """
    if not HAS_GRADIO_CLIENT:
        raise RuntimeError(
            "gradio-client is not installed. Run: pip install gradio-client"
        )

    # ── Prepare temp input file ──────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="reform3d_trellis_")
    tmp_img = os.path.join(tmp_dir, "input.png")
    try:
        _resize_and_save(image_bytes, tmp_img)

        # ── 1. Connect ───────────────────────────────────────────────────────
        logger.info("[Trellis] Connecting to HuggingFace Space...")
        client = _make_client(hf_token)

        # ── 2. Start session ─────────────────────────────────────────────────
        logger.info("[Trellis] Initialising remote session...")
        client.predict(api_name="/start_session")

        # ── 3. Preprocess (background removal on the Space) ──────────────────
        logger.info("[Trellis] Preprocessing image (background removal)...")
        preprocessed = client.predict(
            handle_file(tmp_img),
            api_name="/preprocess_image",
        )
        logger.debug("[Trellis] Preprocessed result: %s", preprocessed)

        # Unwrap dict / tuple result
        if isinstance(preprocessed, dict):
            preprocessed = (
                preprocessed.get("path") or preprocessed.get("url") or tmp_img
            )
        elif isinstance(preprocessed, (list, tuple)):
            first = preprocessed[0]
            preprocessed = (
                first.get("path") or first.get("url")
                if isinstance(first, dict)
                else first
            )

        # ── 4. Tab state ─────────────────────────────────────────────────────
        logger.info("[Trellis] Initialising tab state...")
        client.predict(api_name="/lambda")

        # ── 5. Seed ──────────────────────────────────────────────────────────
        logger.info("[Trellis] Resolving seed...")
        seed_result = client.predict(True, 0, api_name="/get_seed")
        seed = int(seed_result) if seed_result is not None else 0
        logger.info("[Trellis] Seed: %d", seed)

        # ── 6. Generate ──────────────────────────────────────────────────────
        logger.info("[Trellis] Starting 3D generation (1-3 minutes)...")
        image_data = {
            "url": _to_data_uri(preprocessed),
            "meta": {"_type": "gradio.FileData"},
        }
        job = client.submit(
            image_data,   # image
            [],           # multiimages (empty = single-image mode)
            seed,         # seed
            7.5,          # ss_guidance_strength
            12,           # ss_sampling_steps
            3.0,          # slat_guidance_strength
            12,           # slat_sampling_steps
            "stochastic", # multiimage_algo
            0.95,         # mesh_simplify
            1024,         # texture_size
            api_name="/generate_and_extract_glb",
        )

        while not job.done():
            time.sleep(2)

        result = job.result()
        logger.info("[Trellis] Generation complete.")

        # ── 7. Extract GLB path ──────────────────────────────────────────────
        glb_src = _extract_glb_path(result)
        if not glb_src:
            raise RuntimeError(f"TRELLIS returned no GLB file. Raw result: {result}")

        logger.info("[Trellis] GLB source: %s", glb_src)

        # ── 8. Copy to output_dir ────────────────────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_filename = f"trellis_{uuid.uuid4().hex}.glb"
        dest_path = output_dir / dest_filename
        shutil.copy2(glb_src, dest_path)
        logger.info("[Trellis] ✅ Saved GLB to %s", dest_path)
        return dest_path

    finally:
        # Clean up temp input
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
