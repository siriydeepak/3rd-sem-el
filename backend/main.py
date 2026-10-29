import asyncio
import io
import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from PIL import Image

import google.generativeai as genai
from trellis import generate_glb

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
print("Gemini API key loaded:", bool(os.getenv("GEMINI_API_KEY")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The original cloud-masked raster is stored in the old project directory.
RASTER_PATH = (
    Path(r"C:\Users\User\Downloads\3rd-sem-el-main\3rd-sem-el-main")
    / "heatscape_ml"
    / "data"
    / "HeatScape_LST_CloudMasked_Final.tif"
)

app = FastAPI(title="HeatScape API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/")
def read_root():
    return {"message": "HeatScape API v2.1 — colour segmentation active"}


@app.get("/api/reuse-saved-3d")
def reuse_saved_3d():
    saved_model = OUTPUT_DIR / "trellis_8aa5a0093acb4ac0b626ed96fc08be9d.glb"

    if not saved_model.is_file():
        raise HTTPException(status_code=404, detail="Saved 3D model not found.")

    return {
        "status": "success",
        "model_url": f"/outputs/{saved_model.name}",
        "filename": saved_model.name,
    }


# -----------------------------------------------------------------------------
# COLOUR-BASED SEGMENTATION ENGINE
# -----------------------------------------------------------------------------

CLASS_COLOURS = {
    "vegetation": [34, 197, 94],
    "sky": [147, 210, 255],
    "road": [148, 163, 184],
    "building": [249, 115, 22],
    "shadow": [30, 30, 80],
}


def _colour_segment(image_bytes: bytes) -> tuple[dict, bytes, bytes]:
    """Return surface feature percentages, an overlay PNG, and original PNG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    H, W = arr.shape[:2]
    total = H * W
    bright = (R + G + B) / 3.0

    veg = (G > R * 1.08) & (G > B * 1.08) & (G > 55)

    sky = (B > R * 1.05) & (B > G * 0.98) & (B > 100) & ~veg
    sky |= (
        (R > 150)
        & (G > 150)
        & (B > 165)
        & (np.abs(R - G) < 22)
        & (np.abs(G - B) < 22)
        & ~veg
    )

    shadow = bright < 55

    grey_spread = (
        np.abs(R - G) + np.abs(G - B) + np.abs(R - B)
    ).astype(np.float32)
    road = (
        (grey_spread < 38)
        & (bright > 55)
        & (bright < 195)
        & ~veg
        & ~sky
        & ~shadow
    )
    building = ~veg & ~sky & ~road & ~shadow & (bright > 50)

    def pct(mask):
        return round(float(np.sum(mask) / total * 100), 2)

    veg_pct = pct(veg)
    sky_pct = pct(sky)
    road_pct = pct(road)
    bldg_pct = pct(building)
    shadow_pct = pct(shadow)

    alpha = 0.55
    result = arr.copy()
    masks = [
        (veg, CLASS_COLOURS["vegetation"]),
        (sky, CLASS_COLOURS["sky"]),
        (road, CLASS_COLOURS["road"]),
        (building, CLASS_COLOURS["building"]),
        (shadow, CLASS_COLOURS["shadow"]),
    ]
    for mask, colour in masks:
        for channel, colour_value in enumerate(colour):
            result[:, :, channel] = np.where(
                mask,
                arr[:, :, channel] * (1 - alpha) + colour_value * alpha,
                result[:, :, channel],
            )

    overlay_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    ndvi = float(np.mean((G - R) / (G + R + 1e-6)))
    ndbi = float(np.mean((R - B) / (R + B + 1e-6)))
    svf = sky_pct / 100.0

    features = {
        "vegetation_pct": veg_pct,
        "pavement_pct": road_pct,
        "building_pct": bldg_pct,
        "sky_pct": sky_pct,
        "shadow_pct": shadow_pct,
        "sky_view_factor": round(svf, 4),
        "canopy_density": round(veg_pct / 100.0, 4),
        "ndvi": round(ndvi, 4),
        "ndbi": round(ndbi, 4),
        "solar_exposure": round(max(0.0, 1.0 - shadow_pct / 100.0), 4),
        "n_segments": 5,
        "status": "colour_segmentation",
    }

    def to_png(pil_img: Image.Image) -> bytes:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return buffer.getvalue()

    return features, to_png(overlay_img), to_png(img)


@app.post("/api/segment")
async def segment_image(
    file: UploadFile = File(...),
    lat: float = Form(default=0.0),
    lon: float = Form(default=0.0),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    img_id = uuid.uuid4().hex[:10]
    img_ext = Path(file.filename or "photo.jpg").suffix.lower() or ".jpg"
    logger.info("[SEG] Segmenting: %s (%d bytes)", file.filename, len(image_bytes))

    try:
        features, overlay_bytes, original_bytes = await asyncio.to_thread(
            _colour_segment, image_bytes
        )
    except Exception as exc:
        logger.exception("[SEG] Failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}")

    orig_path = UPLOAD_DIR / f"orig_{img_id}.png"
    orig_path.write_bytes(original_bytes)

    overlay_path = UPLOAD_DIR / f"overlay_{img_id}.png"
    overlay_path.write_bytes(overlay_bytes)

    raw_path = UPLOAD_DIR / f"raw_{img_id}{img_ext}"
    raw_path.write_bytes(image_bytes)

    return {
        **features,
        "image_id": img_id,
        "original_url": f"/uploads/{orig_path.name}",
        "overlay_url": f"/uploads/{overlay_path.name}",
        "raw_url": f"/uploads/{raw_path.name}",
        "lat": lat,
        "lon": lon,
    }


@app.post("/api/generate-3d")
async def generate_3d_model(file: UploadFile = File(...)):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN missing in .env")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("[TRELLIS] Generating 3D for: %s", file.filename)
    try:
        glb_path: Path = await asyncio.to_thread(
            generate_glb, image_bytes, HF_TOKEN, OUTPUT_DIR
        )
    except Exception as exc:
        logger.exception("[TRELLIS] Failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"TRELLIS failed: {exc}")

    img_id = uuid.uuid4().hex[:8]
    img_ext = Path(file.filename or "photo.jpg").suffix.lower() or ".jpg"
    img_path = UPLOAD_DIR / f"trellis_src_{img_id}{img_ext}"
    img_path.write_bytes(image_bytes)

    return {
        "status": "success",
        "model_url": f"/outputs/{glb_path.name}",
        "image_url": f"/uploads/{img_path.name}",
        "filename": glb_path.name,
    }


@app.post("/api/interventions")
async def get_interventions(
    file: UploadFile = File(...),
    features_json: str = Form(...),
    budget_inr: float = Form(default=500000),
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in .env")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        features = json.loads(features_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid features_json")

    satellite_context = "Satellite temperature data unavailable."
    try:
        import rasterio

        if RASTER_PATH.exists():
            with rasterio.open(RASTER_PATH) as src:
                raster = src.read(1, masked=True).astype("float32")
                values = raster.compressed()
                values = values[np.isfinite(values)]
                if values.size:
                    satellite_context = json.dumps({
                        "source": RASTER_PATH.name,
                        "units": "Celsius",
                        "mean_c": round(float(np.mean(values)), 2),
                        "min_c": round(float(np.min(values)), 2),
                        "max_c": round(float(np.max(values)), 2),
                    })
    except Exception as exc:
        logger.warning("[SATELLITE] Could not read raster for interventions: %s", exc)

    logger.info("[GEMINI] Analysing image + features for budget ₹%.0f", budget_inr)
    prompt = f"""You are an expert AI Urban Climate Planner specialising in Urban Heat Island (UHI) mitigation.

I have uploaded a ground-level photo of an urban area. Analyse the image carefully — look at the visible surfaces (concrete, trees, roads, buildings, open sky) — and combine that visual understanding with the following automatically extracted data:

Extracted Surface Analysis:
{json.dumps(features, indent=2)}

Available Budget: ₹{budget_inr:,.0f}
Satellite Land Surface Temperature Context: {satellite_context}

Based on BOTH the image and the extracted data, suggest exactly 4 mitigation strategies. Each strategy must:
1. Be specific to what you can SEE in the image
2. Fit within the total budget (strategies should sum to ≤ budget)
3. Include realistic INR cost estimates for Indian cities

Respond ONLY as a valid JSON array — no markdown, no explanation outside the JSON:
[
  {{
    "title": "Short actionable title",
    "reason": "One sentence explaining why this is needed based on the image",
    "what_to_do": "Specific description of the intervention",
    "budget_inr": 150000,
    "temp_reduction_c": 2.1,
    "sustainability_score": 8,
    "implementation_weeks": 4,
    "co_benefits": ["reduces flooding", "improves air quality"]
  }}
]"""

    text = ""
    try:
        model = genai.GenerativeModel("gemini-3.5-flash")
        img_part = {
            "mime_type": file.content_type or "image/jpeg",
            "data": image_bytes,
        }
        response = model.generate_content([prompt, img_part])
        text = response.text.strip()

        if "```" in text:
            for part in text.split("```"):
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("["):
                    text = candidate
                    break

        suggestions = json.loads(text.strip())
        return {"suggestions": suggestions, "total_budget": budget_inr}
    except json.JSONDecodeError as exc:
        logger.error("[GEMINI] JSON parse failed: %s\nRaw: %s", exc, text)
        raise HTTPException(status_code=500, detail="Gemini returned malformed JSON")
    except Exception as exc:
        logger.exception("[GEMINI] Vision call failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------------
# INTERVENTION IMAGE GENERATION
# -----------------------------------------------------------------------------

import httpx

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
print("Stability key loaded:", bool(STABILITY_API_KEY))
print("Stability key length:", len(STABILITY_API_KEY))
FAL_KEY = os.getenv("FAL_KEY", "")


@app.post("/api/generate-intervention-image")
async def generate_intervention_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("[IMG2IMG] Generating intervention image, prompt: %s…", prompt[:60])

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        resized_bytes = buffer.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {exc}")

    full_prompt = (
        "Photorealistic architectural rendering of an urban campus plaza. Same exact camera angle and viewpoint. "
        "Same buildings, existing mature trees, and ground perspective. Added interventions: "
        + prompt.strip().rstrip(".")
        + ". High quality, 8k resolution, crisp architectural details, sunny afternoon lighting."
    )

    if STABILITY_API_KEY:
        try:
            logger.info("[STABILITY] Calling SD3 Image-to-Image…")
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                    headers={
                        "authorization": f"Bearer {STABILITY_API_KEY}",
                        "accept": "image/*",
                    },
                    files={"image": ("input.png", resized_bytes, "image/png")},
                    data={
                        "prompt": full_prompt,
                        "mode": "image-to-image",
                        "strength": "0.60",
                        "model": "sd3-medium",
                        "output_format": "png",
                    },
                )
                if resp.status_code == 200:
                    img_id = uuid.uuid4().hex[:8]
                    out_path = UPLOAD_DIR / f"intervention_{img_id}.png"
                    out_path.write_bytes(resp.content)
                    logger.info("[STABILITY] Success! Saved to %s", out_path.name)
                    return {
                        "status": "success",
                        "image_url": f"/uploads/{out_path.name}",
                        "model": "stability-ai/sd3-medium",
                    }
                logger.warning("[STABILITY] API returned %d: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("[STABILITY] Call failed: %s", exc)

    if FAL_KEY:
        try:
            import base64
            import fal_client

            os.environ["FAL_KEY"] = FAL_KEY
            b64 = base64.b64encode(resized_bytes).decode()
            data_url = f"data:image/png;base64,{b64}"
            result = await asyncio.to_thread(
                fal_client.run,
                "fal-ai/flux/schnell/image-to-image",
                arguments={
                    "image_url": data_url,
                    "prompt": full_prompt,
                    "strength": 0.65,
                    "num_inference_steps": 4,
                    "seed": 42,
                },
            )
            gen_url = result["images"][0]["url"]
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get(gen_url)
                res.raise_for_status()
            img_id = uuid.uuid4().hex[:8]
            out_path = UPLOAD_DIR / f"intervention_{img_id}.png"
            out_path.write_bytes(res.content)
            return {
                "status": "success",
                "image_url": f"/uploads/{out_path.name}",
                "model": "fal-ai/flux-schnell",
            }
        except Exception as exc:
            logger.warning("[FAL] Call failed: %s", exc)

    raise HTTPException(
        status_code=500,
        detail="Image generation failed. Please verify API credits in backend/.env",
    )


# -----------------------------------------------------------------------------
# SATELLITE LAND SURFACE TEMPERATURE ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/satellite/stats")
def satellite_stats():
    import rasterio

    if not RASTER_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Satellite raster not found: {RASTER_PATH}",
        )

    with rasterio.open(RASTER_PATH) as src:
        data = src.read(1, masked=True).astype("float32")
        values = np.asarray(data.compressed())
        values = values[np.isfinite(values)]

        if values.size == 0:
            raise HTTPException(
                status_code=422,
                detail="No valid temperature pixels found.",
            )

        return {
            "status": "success",
            "source": RASTER_PATH.name,
            "units": "Celsius",
            "valid_pixels": int(values.size),
            "mean_c": float(np.mean(values)),
            "min_c": float(np.min(values)),
            "max_c": float(np.max(values)),
            "metadata": {
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
            },
        }


@app.get("/api/satellite/heatmap")
def satellite_heatmap():
    import rasterio
    from rasterio.enums import Resampling

    if not RASTER_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Satellite raster not found: {RASTER_PATH}",
        )

    with rasterio.open(RASTER_PATH) as src:
        width = min(src.width, 1200)
        height = max(1, round(src.height * width / src.width))
        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.average,
        ).astype("float32")

    values = data.compressed()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise HTTPException(status_code=422, detail="No valid temperature pixels")

    low, high = np.percentile(values, [2, 98])
    if high <= low:
        high = low + 1

    raw = data.filled(np.nan)
    invalid = np.ma.getmaskarray(data) | ~np.isfinite(raw)
    normalized = np.clip(
        (np.nan_to_num(raw, nan=low) - low) / (high - low),
        0,
        1,
    )

    stops = np.array([0.0, 0.33, 0.66, 1.0])
    colors = np.array([
        [30, 60, 220],
        [0, 210, 220],
        [255, 230, 0],
        [220, 25, 20],
    ])

    rgba = np.zeros((*raw.shape, 4), dtype=np.uint8)
    for channel in range(3):
        rgba[:, :, channel] = np.interp(
            normalized, stops, colors[:, channel]
        ).astype(np.uint8)
    rgba[:, :, 3] = np.where(invalid, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgba)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

