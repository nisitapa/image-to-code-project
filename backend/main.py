import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv
import base64
import os
import tempfile

try:
    from visual_eval import render_html_to_image, compute_visual_metrics
    print("visual_eval imported successfully", flush=True)
except Exception as e:
    print("visual_eval import failed:", repr(e), flush=True)
    render_html_to_image = None
    compute_visual_metrics = None

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs ของแต่ละ Engine
COLAB_URL = "https://uncombinable-emma-talismanic.ngrok-free.dev/predict"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.get("/")
def read_root():
    return {"status": "Backend is running"}


@app.post("/api/convert/colab/{mode}")
async def convert_colab(mode: str, file: UploadFile = File(...)):
    content = await file.read()
    input_path = None
    generated_img = None
    generated_preview = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(content)
        input_path = tmp.name

    if mode == "body":
        timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=60.0)
    else:
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=60.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {
                "file": (file.filename, content, file.content_type)
            }

            data = {
                "mode": mode
            }

            response = await client.post(
                COLAB_URL,
                files=files,
                data=data
            )

        response.raise_for_status()
        result = response.json()

    except httpx.ReadTimeout:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

        return {
            "mode": mode,
            "html": "",
            "text_similarity": None,
            "color_similarity": None,
            "shape_similarity": None,
            "layout_similarity": None,
            "combined_similarity": None,
            "eval_status": "failed",
            "eval_error": f"Colab timeout while processing mode={mode}"
        }

    except httpx.HTTPStatusError as e:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

        return {
            "mode": mode,
            "html": "",
            "text_similarity": None,
            "color_similarity": None,
            "shape_similarity": None,
            "layout_similarity": None,
            "combined_similarity": None,
            "eval_status": "failed",
            "eval_error": f"Colab HTTP error {e.response.status_code}: {e.response.text}"
        }

    except httpx.RequestError as e:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

        return {
            "mode": mode,
            "html": "",
            "text_similarity": None,
            "color_similarity": None,
            "shape_similarity": None,
            "layout_similarity": None,
            "combined_similarity": None,
            "eval_status": "failed",
            "eval_error": f"Colab connection error: {repr(e)}"
        }

    metrics = {
        "text_similarity": None,
        "color_similarity": None,
        "shape_similarity": None,
        "layout_similarity": None,
        "combined_similarity": None,
        "eval_status": "not_started",
        "eval_error": None
    }

    try:
        html = result.get("html") or result.get("code") or result.get("raw_code")

        if not html:
            raise ValueError(f"No HTML/code found in response. Keys: {list(result.keys())}")

        if render_html_to_image is None or compute_visual_metrics is None:
            metrics["eval_status"] = "visual_eval_unavailable"
            metrics["eval_error"] = "visual_eval import failed or dependency missing"
        else:
            generated_img = await render_html_to_image(html)
            with open(generated_img, "rb") as f:
                generated_preview = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
            calc_metrics = compute_visual_metrics(input_path, generated_img, mode=mode)

            metrics.update(calc_metrics)
            metrics["eval_status"] = "success"
            metrics["eval_error"] = None

            print("===== ONLINE VISUAL EVAL =====", flush=True)
            print("Engine: Colab", flush=True)
            print("Mode:", mode, flush=True)
            print("Status:", metrics["eval_status"], flush=True)
            if metrics["eval_error"]:
                print("Error:", metrics["eval_error"], flush=True)
            print("Text Similarity:", metrics["text_similarity"], flush=True)
            print("Color Similarity:", metrics["color_similarity"], flush=True)
            print("Shape Similarity:", metrics["shape_similarity"], flush=True)
            print("Layout Similarity:", metrics["layout_similarity"], flush=True)
            print("Combined Similarity:", metrics["combined_similarity"], flush=True)
            print("===============================", flush=True)

    except Exception as e:
        metrics["eval_status"] = "failed"
        metrics["eval_error"] = repr(e)
        print("Evaluation error:", repr(e), flush=True)

    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

        if generated_img and os.path.exists(generated_img):
            try:
                os.remove(generated_img)
            except Exception:
                pass

    return {
        **result,
        **metrics,
        "generated_preview": generated_preview
    }

@app.post("/api/convert/gemini/{mode}")
async def convert_gemini(mode: str, file: UploadFile = File(...)):
    content = await file.read()
    base64_img = base64.b64encode(content).decode("utf-8")

    input_path = None
    generated_img = None
    generated_preview = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(content)
        input_path = tmp.name

    json_payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": file.content_type,
                            "data": base64_img
                        }
                    },
                    {
                        "text": """You are an expert frontend developer.
Convert this UI image into a single HTML file using INLINE CSS ONLY.
All CSS must be in style="" attributes.
Do NOT use <style> tags, classes, IDs, or external stylesheets.
Return ONLY valid HTML with <!DOCTYPE html>, <html>, <head>, <body> and proper closed tags.
Do NOT explain anything, do not add extra text, do not truncate."""
                    }
                ]
            }
        ]
    }

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    metrics = {
        "text_similarity": None,
        "color_similarity": None,
        "shape_similarity": None,
        "layout_similarity": None,
        "combined_similarity": None,
        "eval_status": "not_started",
        "eval_error": None
    }

    html_content = ""

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(GEMINI_API_URL, headers=headers, json=json_payload)
            response.raise_for_status()
            data = response.json()

        html_content = data["candidates"][0]["content"]["parts"][0]["text"]

        if html_content.startswith("```html") and html_content.endswith("```"):
            html_content = html_content[7:-3].strip()

        if render_html_to_image is None or compute_visual_metrics is None:
            metrics["eval_status"] = "visual_eval_unavailable"
            metrics["eval_error"] = "visual_eval import failed or dependency missing"
        else:
            generated_img = await render_html_to_image(html_content)
            with open(generated_img, "rb") as f:
                generated_preview = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
            calc_metrics = compute_visual_metrics(input_path, generated_img, mode=mode)

            metrics.update(calc_metrics)
            metrics["eval_status"] = "success"
            metrics["eval_error"] = None

        print("===== ONLINE VISUAL EVAL =====", flush=True)
        print("Engine: Gemini", flush=True)
        print("Mode:", mode, flush=True)
        print("Status:", metrics["eval_status"], flush=True)
        if metrics["eval_error"]:
            print("Error:", metrics["eval_error"], flush=True)
        print("Text Similarity:", metrics["text_similarity"], flush=True)
        print("Color Similarity:", metrics["color_similarity"], flush=True)
        print("Shape Similarity:", metrics["shape_similarity"], flush=True)
        print("Layout Similarity:", metrics["layout_similarity"], flush=True)
        print("Combined Similarity:", metrics["combined_similarity"], flush=True)
        print("===============================", flush=True)

        return {
            "html": html_content,
            **metrics,
            "generated_preview": generated_preview
        }

    except Exception as e:
        metrics["eval_status"] = "failed"
        metrics["eval_error"] = repr(e)
        print("Evaluation error:", repr(e), flush=True)

        return {
            "html": html_content,
            **metrics
        }

    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

        if generated_img and os.path.exists(generated_img):
            try:
                os.remove(generated_img)
            except Exception:
                pass