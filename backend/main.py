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
from PIL import Image

import google.generativeai as genai
from trellis import generate_glb
from exif_service import extract_exif_metadata
from satellite_weather_service import get_satellite_weather
from heat_model_service import evaluate_interventions
from heatmap_service import generate_urban_heat_map
from semantic_thermal_service import get_semantic_temperatures, get_vertex_color_mapping

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN       = os.getenv("HF_TOKEN")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="HeatScape API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)),  name="uploads")


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR-BASED SEGMENTATION ENGINE
#  Works immediately — no model download needed.
#  Uses HSV colour analysis on the image to classify urban surfaces.
#  When SAM2 checkpoint is available, replace this with real mask generation.
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLOURS = {
    "vegetation" : [34,  197,  94],   # green
    "sky"        : [147, 210, 255],   # light blue
    "road"       : [148, 163, 184],   # slate grey
    "building"   : [249, 115,  22],   # orange
    "shadow"     : [ 30,  30,  80],   # near black
}

def _colour_segment(image_bytes: bytes) -> tuple[dict, bytes, bytes]:
    """
    Fast colour-based urban surface segmentation using PIL + NumPy.

    Returns
    -------
    features     : dict of percentages, SVF, NDVI, NDBI
    overlay_png  : PNG bytes of the original image with coloured overlays
    original_png : PNG bytes of the resized original (for the report)
    """
    img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Cap at 1024px on the long side for speed
    img.thumbnail((1024, 1024), Image.LANCZOS)
    arr     = np.array(img, dtype=np.float32)

    R, G, B  = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    H, W     = arr.shape[:2]
    total    = H * W
    bright   = (R + G + B) / 3.0

    # ── Class masks ───────────────────────────────────────────────────────────
    # Vegetation: green dominates red and blue
    veg   = (G > R * 1.08) & (G > B * 1.08) & (G > 55)

    # Sky: blue > others, relatively bright & unsaturated
    sky   = (B > R * 1.05) & (B > G * 0.98) & (B > 100) & ~veg
    # Overcast / white sky
    sky  |= (R > 150) & (G > 150) & (B > 165) & \
            (np.abs(R - G) < 22) & (np.abs(G - B) < 22) & ~veg

    # Shadow / very dark pixels
    shadow = bright < 55

    # Road / pavement: grey (R≈G≈B), mid brightness
    grey_spread = (np.abs(R - G) + np.abs(G - B) + np.abs(R - B)).astype(np.float32)
    road  = (grey_spread < 38) & (bright > 55) & (bright < 195) & ~veg & ~sky & ~shadow

    # Buildings: everything remaining above shadow threshold
    building = ~veg & ~sky & ~road & ~shadow & (bright > 50)

    # ── Percentages ───────────────────────────────────────────────────────────
    def pct(m): return round(float(np.sum(m) / total * 100), 2)
    veg_pct    = pct(veg)
    sky_pct    = pct(sky)
    road_pct   = pct(road)
    bldg_pct   = pct(building)
    shadow_pct = pct(shadow)

    # ── Coloured overlay (blend 55% colour, 45% original) ────────────────────
    ALPHA   = 0.55
    result  = arr.copy()
    masks   = [
        (veg,      CLASS_COLOURS["vegetation"]),
        (sky,      CLASS_COLOURS["sky"]),
        (road,     CLASS_COLOURS["road"]),
        (building, CLASS_COLOURS["building"]),
        (shadow,   CLASS_COLOURS["shadow"]),
    ]
    for mask, colour in masks:
        for ch, cv in enumerate(colour):
            result[:, :, ch] = np.where(mask, arr[:, :, ch] * (1 - ALPHA) + cv * ALPHA, result[:, :, ch])

    overlay_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    # ── Spectral indices (RGB proxies) ────────────────────────────────────────
    ndvi = float(np.mean((G - R) / (G + R + 1e-6)))
    ndbi = float(np.mean((R - B) / (R + B + 1e-6)))
    svf  = sky_pct / 100.0

    features = {
        "vegetation_pct"  : veg_pct,
        "pavement_pct"    : road_pct,
        "building_pct"    : bldg_pct,
        "sky_pct"         : sky_pct,
        "shadow_pct"      : shadow_pct,
        "sky_view_factor" : round(svf, 4),
        "canopy_density"  : round((veg_pct) / 100.0, 4),
        "ndvi"            : round(ndvi, 4),
        "ndbi"            : round(ndbi, 4),
        "solar_exposure"  : round(max(0.0, 1.0 - shadow_pct / 100.0), 4),
        "n_segments"      : 5,
        "status"          : "colour_segmentation",
    }

    # Serialise images to PNG bytes
    def _to_png(pil_img: Image.Image) -> bytes:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    return features, _to_png(overlay_img), _to_png(img)


# ─────────────────────────────────────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "HeatScape API v2.1 — colour segmentation active"}


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — /api/segment
#  Receives the uploaded image and returns:
#    • JSON feature stats
#    • overlay_url — the original image with coloured surface overlays
#    • original_url — resized original saved for the report
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/segment")
async def segment_image(
    file: UploadFile = File(...),
    lat:  float = Form(default=0.0),
    lon:  float = Form(default=0.0),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    img_id  = uuid.uuid4().hex[:10]
    img_ext = Path(file.filename or "photo.jpg").suffix.lower() or ".jpg"

    logger.info("[SEG] Segmenting: %s (%d bytes)", file.filename, len(image_bytes))

    # 1. Surface Segmentation
    try:
        features, overlay_bytes, original_bytes = await asyncio.to_thread(
            _colour_segment, image_bytes
        )
    except Exception as exc:
        logger.exception("[SEG] Failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}")

    # 2. Extract EXIF Geotag & Capture Timestamp
    exif_meta = extract_exif_metadata(image_bytes)
    resolved_lat = lat if (lat != 0.0) else exif_meta["latitude"]
    resolved_lon = lon if (lon != 0.0) else exif_meta["longitude"]

    # 3. Retrieve Satellite & Ambient Environmental Thermal Data
    weather = await get_satellite_weather(
        resolved_lat, resolved_lon, exif_meta.get("timestamp")
    )

    # 4. Connect to HeatScape ML Model for Localized LST & Interventions
    combined_features = {
        **features,
        "air_temperature": weather["air_temperature"],
        "humidity": weather["humidity"],
        "solar_exposure": weather["solar_exposure"],
        "satellite_lst": weather["satellite_lst"],
        "tree_canopy_pct": round(features.get("vegetation_pct", 0) * 0.7, 2),
    }

    heat_analysis = evaluate_interventions(
        combined_features,
        air_temp=weather["air_temperature"],
        sat_lst=weather["satellite_lst"]
    )

    # 5. Generate Urban Heat Map Grid
    heatmap_data = generate_urban_heat_map(
        center_lat=resolved_lat,
        center_lon=resolved_lon,
        baseline_lst=heat_analysis["baseline_lst"],
        mitigated_drop=heat_analysis["interventions"]["combined"]["temp_reduction_c"],
        radius_meters=300.0,
        grid_size=5
    )

    # Save original (resized)
    orig_path = UPLOAD_DIR / f"orig_{img_id}.png"
    orig_path.write_bytes(original_bytes)

    # Save coloured overlay
    ovly_path = UPLOAD_DIR / f"overlay_{img_id}.png"
    ovly_path.write_bytes(overlay_bytes)

    # Also save raw original with original extension (for TRELLIS)
    raw_path  = UPLOAD_DIR / f"raw_{img_id}{img_ext}"
    raw_path.write_bytes(image_bytes)

    # 6. Semantic Thermal Heuristics (Member 3: Object-level mapping & 3D vertex colors)
    semantic_thermal = get_semantic_temperatures(
        heat_analysis["baseline_lst"],
        solar_exposure=weather.get("solar_exposure", 0.85)
    )
    vertex_colors = get_vertex_color_mapping(heat_analysis["baseline_lst"])

    return {
        **features,
        "image_id"         : img_id,
        "original_url"     : f"/uploads/{orig_path.name}",
        "overlay_url"      : f"/uploads/{ovly_path.name}",
        "raw_url"          : f"/uploads/{raw_path.name}",
        "lat"              : resolved_lat,
        "lon"              : resolved_lon,
        "exif"             : exif_meta,
        "weather"          : weather,
        "heat_analysis"    : heat_analysis,
        "heatmap"          : heatmap_data,
        "semantic_thermal" : semantic_thermal,
        "vertex_colors"    : vertex_colors,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.5 — /api/heatmap (Dynamic microclimate heat map recalculation)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/heatmap")
async def get_heatmap_custom(
    lat: float = Form(...),
    lon: float = Form(...),
    baseline_lst: float = Form(...),
    mitigated_drop: float = Form(default=3.5),
    radius_meters: float = Form(default=300.0)
):
    """
    Returns localized heat map points based on coordinates, baseline LST, and intervention drop.
    """
    return generate_urban_heat_map(
        center_lat=lat,
        center_lon=lon,
        baseline_lst=baseline_lst,
        mitigated_drop=mitigated_drop,
        radius_meters=radius_meters,
        grid_size=5
    )


@app.get("/api/semantic-thermal")
async def get_semantic_thermal_info(baseline_lst: float = 38.0, solar_exposure: float = 0.85):
    """
    Returns semantic object temperatures and Three.js vertex color mappings for 3D model coloring.
    """
    return {
        "baseline_lst": baseline_lst,
        "semantic_temperatures": get_semantic_temperatures(baseline_lst, solar_exposure),
        "vertex_colors": get_vertex_color_mapping(baseline_lst)
    }



# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — /api/generate-3d  (TRELLIS — exact copy from HACK4SDG)
# ─────────────────────────────────────────────────────────────────────────────

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

    img_id  = uuid.uuid4().hex[:8]
    img_ext = Path(file.filename or "photo.jpg").suffix.lower() or ".jpg"
    img_path = UPLOAD_DIR / f"trellis_src_{img_id}{img_ext}"
    img_path.write_bytes(image_bytes)

    return {
        "status"    : "success",
        "model_url" : f"/outputs/{glb_path.name}",
        "image_url" : f"/uploads/{img_path.name}",
        "filename"  : glb_path.name,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — /api/interventions
#  Accepts the uploaded image + segmentation features + budget.
#  Gemini Vision analyses the ACTUAL image and returns structured interventions
#  calibrated to the user's budget.
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/interventions")
async def get_interventions(
    file          : UploadFile = File(...),
    features_json : str        = Form(...),    # JSON string of segmentation features
    budget_inr    : float      = Form(default=500000),
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

    logger.info("[GEMINI] Analysing image + features for budget ₹%.0f", budget_inr)

    prompt = f"""You are an expert AI Urban Climate Planner specialising in Urban Heat Island (UHI) mitigation.

I have uploaded a ground-level photo of an urban area. Analyse the image carefully — look at the visible surfaces (concrete, trees, roads, buildings, open sky) — and combine that visual understanding with the following automatically extracted data:

Extracted Surface Analysis:
{json.dumps(features, indent=2)}

Available Budget: ₹{budget_inr:,.0f}

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

    try:
        model = genai.GenerativeModel("gemini-3.5-flash")

        # Use Gemini Vision — send the actual image
        img_part = {"mime_type": file.content_type or "image/jpeg", "data": image_bytes}
        response = model.generate_content([prompt, img_part])

        text = response.text.strip()
        # Strip markdown fences if present
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("["):
                    text = p
                    break

        suggestions = json.loads(text.strip())
        return {"suggestions": suggestions, "total_budget": budget_inr}

    except json.JSONDecodeError as exc:
        logger.error("[GEMINI] JSON parse failed: %s\nRaw: %s", exc, text)
        raise HTTPException(status_code=500, detail="Gemini returned malformed JSON")
    except Exception as exc:
        logger.exception("[GEMINI] Vision call failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — /api/generate-intervention-image
#  Generates photorealistic urban intervention visualization using Image-to-Image.
#  Uses Stability AI SD3 (active credits) with structure preservation.
# ─────────────────────────────────────────────────────────────────────────────

import httpx

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
FAL_KEY = os.getenv("FAL_KEY", "")

@app.post("/api/generate-intervention-image")
async def generate_intervention_image(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("[IMG2IMG] Generating intervention image, prompt: %s…", prompt[:60])

    # Resize to max 1024x1024 for fast and reliable API processing
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        resized_bytes = buf.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {exc}")

    full_prompt = (
        "Photorealistic architectural rendering of an urban campus plaza. Same exact camera angle and viewpoint. "
        "Same buildings, existing mature trees, and ground perspective. Added interventions: "
        + prompt.strip().rstrip(".") + ". "
        "High quality, 8k resolution, crisp architectural details, sunny afternoon lighting."
    )

    # 1. Try Stability AI SD3 Image-to-Image (active with confirmed credits)
    if STABILITY_API_KEY:
        try:
            logger.info("[STABILITY] Calling SD3 Image-to-Image…")
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                    headers={
                        "authorization": f"Bearer {STABILITY_API_KEY}",
                        "accept": "image/*"
                    },
                    files={"image": ("input.png", resized_bytes, "image/png")},
                    data={
                        "prompt": full_prompt,
                        "mode": "image-to-image",
                        "strength": 0.60,
                        "model": "sd3-medium",
                        "output_format": "png"
                    }
                )

                if resp.status_code == 200:
                    img_id = uuid.uuid4().hex[:8]
                    out_path = UPLOAD_DIR / f"intervention_{img_id}.png"
                    out_path.write_bytes(resp.content)
                    logger.info("[STABILITY] Success! Saved to %s", out_path.name)
                    return {
                        "status": "success",
                        "image_url": f"/uploads/{out_path.name}",
                        "model": "stability-ai/sd3-medium"
                    }
                else:
                    logger.warning("[STABILITY] API returned %d: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("[STABILITY] Call failed: %s", exc)

    # 2. Try fal.ai if configured and active
    if FAL_KEY:
        try:
            import fal_client
            os.environ["FAL_KEY"] = FAL_KEY
            import base64
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
                    "seed": 42
                }
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
                "model": "fal-ai/flux-schnell"
            }
        except Exception as exc:
            logger.warning("[FAL] Call failed: %s", exc)

    raise HTTPException(
        status_code=500,
        detail="Image generation failed. Please verify API credits in backend/.env"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

