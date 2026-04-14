# =========================================================
# Mimic Inference API Colab
# FastAPI server for UI image -> HTML generation
# =========================================================

# =========================
# INSTALL
# =========================
# Run these manually in Colab if needed:
# !pip install --quiet torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
# !pip install --quiet fsspec==2025.3.0
# !pip install --quiet transformers peft fastapi uvicorn bs4 pyngrok bitsandbytes>=0.46.1

# =========================
# IMPORTS
# =========================
from pyngrok import ngrok
import uvicorn
import threading
import io
import gc
import torch
import re
import os

from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
from google.colab import drive

# =========================
# MOUNT DRIVE
# =========================
drive.mount("/content/drive", force_remount=False)

# =========================
# NGROK
# =========================
NGROK_AUTHTOKEN = "YOUR_NGROK_TOKEN"

!ngrok config add-authtoken $NGROK_AUTHTOKEN
public_url = ngrok.connect(8000).public_url
print("Public URL:", public_url)

# =========================
# MODEL CONFIG
# =========================
base_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

ADAPTERS = {
    "header": "/content/drive/MyDrive/UI_Dataset/Model/model_Header",
    "footer": "/content/drive/MyDrive/UI_Dataset/Model/model_Footer",
    "body": "/content/drive/MyDrive/UI_Dataset/Model/modeled/model_Body3",
    "full": "/content/drive/MyDrive/UI_Dataset/Model/modeled/model_Full2",
    "login": "/content/drive/MyDrive/UI_Dataset/Model/model_Login",
    "regist": "/content/drive/MyDrive/UI_Dataset/Model/model_Regist",
    "forget": "/content/drive/MyDrive/UI_Dataset/Model/model_Forget",
}

PROMPTS = {
    "header": """[INST]
<image>

Convert the given UI image into EXACTLY ONE HTML <header> component.

Rules:
- Output ONLY one HTML fragment
- Root element MUST be <header>
- Use INLINE CSS ONLY in style=""
- Do NOT output markdown, explanation, <html>, <head>, <body>, <style>, class, or id
- Preserve layout, spacing, alignment, colors, sizes, and typography as closely as possible
- Include all visible header elements only
- Do not invent missing elements
- If all text items are one centered navigation row, keep them in one centered row
- Do not separate a logo/title unless it is clearly visually separated
- Start with <header style="...">
- End with </header>

Return ONLY the final HTML fragment.
[/INST]""",

    "footer": """[INST]
<image>

Convert ONLY the visible footer area from the UI image into ONE HTML fragment.

Rules:
- Output HTML only
- Use inline CSS only with style=""
- Do not use <style>, class, id, <html>, <head>, or <body>
- Reproduce only visible footer elements
- Copy only clearly visible text
- Preserve layout, spacing, alignment, colors, and typography as closely as possible
- Do not invent extra text, icons, logos, images, links, columns, or sections
- If the footer contains one visible text line, keep it as one line
- If multiple footer items are arranged horizontally, preserve the horizontal layout
- Keep the structure minimal and close to the image

Required structure:
<div style="min-height:100vh;display:flex;flex-direction:column;">
<div style="flex:1;"></div>
<footer style="width:100%;">
...
</footer>
</div>

Return only the final HTML fragment.
[/INST]""",

    "body": """[INST]
<image>

You are a STRICT UI-to-HTML BODY reconstruction engine.

Task:
Convert the visible BODY area into EXACTLY ONE HTML fragment.

Hard rules:
- Output ONLY one HTML fragment
- Root element MUST be <div>
- Use INLINE CSS ONLY in style=""
- EVERY tag MUST contain style=""
- Do NOT output markdown, explanation, <html>, <head>, <body>, <style>, class, or id
- Reproduce ONLY visible elements
- Do NOT invent missing elements
- Do NOT omit visible elements
- Do NOT create deeply nested empty div chains
- Do NOT create recursive container patterns

Priority order:
1. First preserve the global layout structure
2. Then preserve relative positions and spacing
3. Then preserve colors, backgrounds, borders, and image regions
4. Then preserve text and button labels

Layout rules:
- Preserve the exact number of major sections
- Preserve row/column relationships exactly as visible
- Preserve card sizes, card positions, and gaps between cards
- Preserve image areas as separate blocks inside cards
- Preserve bottom split sections if visible
- Do NOT collapse multiple visible cards into one
- Build the BODY from large visible layout regions first, then inner content

Color rules:
- Match background colors, card colors, border colors, and button colors as closely as possible
- Use HEX colors only
- Do NOT replace visible colored regions with plain white unless they are clearly white
- Preserve contrast differences between neighboring sections

Text rules:
- Include readable text
- Preserve approximate typography, alignment, and button placement
- Do not return text-only HTML

Structure Rules:
- Output MUST start with <div style="...">
- Output MUST end with </div>

Return ONLY the final HTML fragment.
[/INST]""",

    "full": """[INST]
<image>
Convert the full visible webpage in the screenshot into a single HTML fragment using INLINE CSS ONLY.

Your task is to reconstruct the page layout from the screenshot itself.

REQUIREMENTS:
- Recreate the complete visible page from top to bottom
- Use a single root <div>
- Preserve all major visible sections in order
- Keep visually distinct sections separate
- Use appropriate HTML elements for visible buttons, inputs, links, cards, images, headings, and paragraphs

RULES:
- INLINE CSS ONLY
- Do NOT use <html>, <head>, <body>, <style>, <script>, class, or id
- Use only HEX color codes
- Do NOT invent new sections or content that are not visible
- Do NOT collapse a multi-section page into one banner or one text block
- Do NOT omit visible bottom regions such as content sections, forms, or footers

PRIORITY:
1. Overall page structure
2. Section separation
3. Element placement and spacing
4. Visual styling
5. Text content

If text is unclear, use short neutral placeholder text while preserving layout.

Output ONLY the final HTML fragment, starting with one root <div>.
[/INST]""",

    "login": """[INST]
<image>

You are a STRICT UI-to-HTML AUTH PAGE conversion engine.
This is a deterministic visual reconstruction task, NOT a design task.

PRIMARY GOAL:
Reconstruct the visible login/sign-up UI from the image with maximum visual accuracy.

ABSOLUTE OUTPUT RULES:
- Output HTML only
- Use exactly ONE root <div>
- Use inline CSS only with style=""
- EVERY HTML tag must contain style=""
- Do NOT use <html>, <head>, <body>, <style>, class, or id
- Use HEX colors only
- Reproduce ONLY what is clearly visible in the image
- Do NOT add any element that is not clearly visible

CRITICAL VISUAL RULES:
- Preserve the exact visible composition, alignment, spacing, and relative sizing
- Preserve the exact visible number of text lines, inputs, and buttons
- Preserve whether the layout is top-aligned, left-aligned, centered, wide, narrow, or full-width
- Preserve the background color of the page as closely as possible
- Preserve the visible colors of buttons, inputs, text, and borders as closely as possible
- Preserve border radius, padding, typography, and element width/height as closely as possible
- If the content appears directly on the page background, keep it directly on the page background
- Use a card, panel, modal, or separate container only if it is clearly visible in the image
- Match the visible composition from the image, including whether the layout appears centered or not
- Prefer the specific visible colors in the image over common default web UI colors
- Do NOT invent shadows, borders, helper text, password fields, links, logos, icons, dividers, or secondary actions unless clearly visible

TEXT RULES:
- Copy clearly visible text as closely as possible
- Keep text casing consistent with the image
- Preserve visible line breaks and label placement

STRUCTURE RULES:
- If the title, label, input, and button are stacked vertically in the image, keep them vertically stacked
- If elements are left-aligned in the image, keep them left-aligned
- If an input or button visually spans most of the content width, reflect that width in HTML
- Preserve the overall visible scale of the layout rather than shrinking it into a generic form

FINAL VALIDATION:
- One root <div> only
- No extra containers not visible in the image
- No generic auth redesign
- The result should visually resemble the image, not a common login template

Return only the final HTML fragment.
[/INST]""",

    "regist": """[INST]
<image>

You are a UI-to-HTML conversion engine.

Task:
Convert the given UI image into ONE standalone register/sign-up page HTML fragment.

Rules:
- Output HTML only
- Use exactly ONE root <div>
- Use inline CSS only with style=""
- Every HTML tag must contain style=""
- Do not use <html>, <head>, <body>, <style>, class, or id
- Reproduce only elements that are clearly visible in the image
- Preserve the visible layout direction, spacing, and relative sizing
- If elements are stacked vertically in the image, keep them vertically stacked
- If elements are side-by-side in the image, keep them side-by-side
- Preserve the exact number of visible input fields, buttons, and text blocks
- Include only buttons and helper text that are clearly visible in the image
- Copy clearly visible text as closely as possible
- Use the visible input placeholder text directly when it is readable
- Match colors, border radius, typography, width, height, padding, and alignment as closely as possible
- Do not add extra fields, labels, checkboxes, password fields, icons, logos, or links unless they are clearly visible in the image
- Do not add a separate label above the input if the input already shows visible placeholder text
- Do not rewrite placeholder text into longer form text
- Do not replace the visible design with a generic multi-field register template
- Do not invent extra sections or controls that are commonly found in sign-up pages but are not visible here
- Do not add secondary or step-navigation buttons unless clearly visible
- Do not invent actions such as Next, Continue, Submit, or Log In unless clearly visible
- Keep all visible register elements inside one unified layout block when they visually belong together
- Do not split bottom helper text into separate lines unless clearly visible
- Do not center the title if it is visually left-aligned in the image
- Use HEX colors only

Return only the final HTML fragment.
[/INST]""",

    "forget": """[INST]
<image>

Convert the UI image into a FORGOT PASSWORD page.

Rules:
- Output ONLY HTML
- Use ONE root <div>
- Use INLINE CSS only (style="")
- Do NOT use <html>, <head>, <body>, class, or id
- Include only elements visible in the image
- Colors should match the UI image as closely as possible using HEX color codes

Return the HTML fragment only.
[/INST]""",
}

BODY_RETRY_PROMPT = """[INST]
<image>

Convert the visible BODY into ONE HTML fragment.

Strict rules:
- Root must be one <div>
- Inline CSS only
- Every tag must contain style=""
- Reconstruct large layout regions first, then inner content
- Do NOT create deeply nested empty div chains
- Do NOT create recursive container patterns
- Preserve the visible large cards/tiles first
- Preserve text blocks, buttons, and image areas inside each card
- Preserve shapes, blocks, panels, and borders as visible visual regions
- If a 2-column card grid is visible, keep that grid
- If a bottom horizontal split section is visible, keep it
- If a card contains an image, represent it as a separate visible block
- Do not return text-only HTML
- Do not invent extra elements

Return only HTML.
[/INST]"""

BODY_LAYOUT_COLOR_RETRY_PROMPT = """[INST]
<image>

Rebuild the BODY into ONE HTML fragment.

Strict requirements:
- Root must be one <div>
- Inline CSS only
- Every tag must contain style=""
- Focus on matching the LARGE layout first
- Focus on card positions, section alignment, gaps, widths, heights, and backgrounds
- Focus on matching visible colors, border contrast, radii, and section contrast
- Preserve image regions, decorative blocks, and geometric shapes as distinct visual blocks
- Preserve button positions inside each card
- Do NOT simplify the page into a text-first layout
- Do NOT use repetitive nested empty containers
- If a 2-column card layout is visible, preserve it exactly
- If a bottom horizontal split section is visible, preserve it exactly
- If cards or panels have different fills, preserve those differences
- If visible blocks have borders or rounded corners, preserve them

Return only HTML.
[/INST]"""

BODY_SHAPE_COLOR_RETRY_PROMPT = """[INST]
<image>

Reconstruct the BODY as a visual layout.

Rules:
- Output one root <div> only
- Inline CSS only
- Every tag must contain style=""
- Prioritize visible shapes, blocks, cards, image placeholders, borders, rounded corners, and background fills
- Preserve relative positions and spacing between blocks
- Preserve section/background color differences
- Preserve card/background color differences
- Preserve button fill colors and block fills
- Preserve visible rectangular image/media areas as separate blocks
- Keep readable text, but do not let text dominate over layout reconstruction
- Do NOT simplify distinct visual regions into plain stacked text blocks
- Do NOT invent extra content

Return only HTML.
[/INST]"""

# =========================
# LOAD PROCESSOR + BASE MODEL
# =========================
processor = LlavaNextProcessor.from_pretrained(base_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

gc.collect()
torch.cuda.empty_cache()

base_model = LlavaNextForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
).eval()

# =========================
# ADAPTER CACHE
# =========================
MODEL_CACHE = {}

def get_model(mode: str):
    if mode not in ADAPTERS:
        raise ValueError(f"Invalid mode: {mode}")

    if mode not in MODEL_CACHE:
        print("Loading adapter:", mode)
        MODEL_CACHE[mode] = PeftModel.from_pretrained(
            base_model,
            ADAPTERS[mode],
        ).eval()

    return MODEL_CACHE[mode]

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

def resize_image(image, max_side=1024):
    w, h = image.size
    scale = min(max_side / w, max_side / h, 1.0)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def resize_image_header_aware(image, mode: str):
    if mode != "header":
        return resize_image(image)

    w, h = image.size
    aspect = w / max(h, 1)

    if aspect >= 2.2:
        return resize_image(image, max_side=1408)
    elif aspect >= 1.7:
        return resize_image(image, max_side=1280)
    else:
        return resize_image(image, max_side=1024)

def resize_image_footer_aware(image, mode: str):
    if mode == "footer":
        w, h = image.size
        aspect = w / max(h, 1)
        area = w * h

        if aspect >= 4.5 or area >= 1400000:
            return resize_image(image, max_side=1536)
        elif aspect >= 3.0 or area >= 900000:
            return resize_image(image, max_side=1408)
        elif aspect >= 2.0:
            return resize_image(image, max_side=1280)
        else:
            return resize_image(image, max_side=1024)
    return image

def resize_image_body_aware(image, mode: str):
    if mode != "body":
        return image

    w, h = image.size
    aspect = h / max(w, 1)
    area = w * h

    if aspect >= 1.8 or area >= 1600000:
        return resize_image(image, max_side=1408)
    elif aspect >= 1.2 or area >= 1000000:
        return resize_image(image, max_side=1280)
    else:
        return resize_image(image, max_side=1152)

def get_body_params(image):
    w, h = image.size
    aspect = h / max(w, 1)
    area = w * h

    if aspect >= 1.8 or area >= 1600000:
        return 1200, 1.00
    elif aspect >= 1.2 or area >= 1000000:
        return 1120, 1.00
    else:
        return 1080, 1.00

def get_header_params(image):
    w, h = image.size
    aspect = w / max(h, 1)
    area = w * h

    if area >= 1200000 or aspect >= 2.2:
        return 900, 1.02
    elif area >= 700000 or aspect >= 1.7:
        return 720, 1.02
    else:
        return 520, 1.02

def extract_balanced_root(text: str, tag_name: str) -> str:
    pattern = rf"</?{tag_name}\b[^>]*>"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    if not matches:
        return text.strip()

    open_count = 0
    start_pos = None
    end_pos = None

    for m in matches:
        tag = m.group(0).lower()

        if start_pos is None and not tag.startswith(f"</{tag_name}"):
            start_pos = m.start()

        if tag.startswith(f"</{tag_name}"):
            open_count -= 1
        else:
            open_count += 1

        if start_pos is not None and open_count == 0:
            end_pos = m.end()
            break

    if start_pos is not None and end_pos is not None:
        return text[start_pos:end_pos].strip()

    return text.strip()

def clean_html_header(raw_html):
    try:
        if isinstance(raw_html, list):
            raw_html = " ".join(str(x) for x in raw_html)

        text = raw_html.replace("\\n", "\n")
        text = re.sub(r"\[INST\].*?\[/INST\]", "", text, flags=re.DOTALL).strip()

        first_tag = re.search(r"<header\b", text, flags=re.IGNORECASE)
        if first_tag:
            text = text[first_tag.start():]

        text = extract_balanced_root(text, "header")

        if text.lower().startswith("<header") and "</header>" not in text.lower():
            text += "</header>"

        return text.strip()

    except Exception as e:
        print("HEADER CLEAN ERROR:", e)
        return raw_html

def clean_html(raw_html):
    try:
        if isinstance(raw_html, list):
            raw_html = " ".join(str(x) for x in raw_html)

        text = raw_html.replace("\\n", "\n")
        text = re.sub(r"\[INST\].*?\[/INST\]", "", text, flags=re.DOTALL).strip()

        first_tag = re.search(r"<(div|header|footer|body)\b", text, flags=re.IGNORECASE)
        if first_tag:
            text = text[first_tag.start():]

        if text.startswith("<div"):
            matches = list(re.finditer(r"</?div\b[^>]*>", text, flags=re.IGNORECASE))
            open_divs = 0
            end_pos = None

            for m in matches:
                tag = m.group(0).lower()
                if tag.startswith("</div"):
                    open_divs -= 1
                else:
                    open_divs += 1

                if open_divs == 0:
                    end_pos = m.end()
                    break

            if end_pos:
                text = text[:end_pos]

        elif text.startswith("<body"):
            matches = list(re.finditer(r"</?body\b[^>]*>", text, flags=re.IGNORECASE))
            open_bodies = 0
            end_pos = None

            for m in matches:
                tag = m.group(0).lower()
                if tag.startswith("</body"):
                    open_bodies -= 1
                else:
                    open_bodies += 1

                if open_bodies == 0:
                    end_pos = m.end()
                    break

            if end_pos:
                text = text[:end_pos]

        return text.strip()

    except Exception as e:
        print("CLEAN ERROR:", e)
        return raw_html

def sanitize_login_html(html: str) -> str:
    html = re.sub(r'<input[^>]*type="hidden"[^>]*\/?>', "", html, flags=re.IGNORECASE)
    html = re.sub(r"<input[^>]*type='hidden'[^>]*\/?>", "", html, flags=re.IGNORECASE)
    html = re.sub(r'<input[^>]*type="checkbox"[^>]*\/?>', "", html, flags=re.IGNORECASE)
    html = re.sub(r"<input[^>]*type='checkbox'[^>]*\/?>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<label[^>]*>\s*Remember me\s*</label>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<a[^>]*>\s*Forgot password\?\s*</a>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<span[^>]*>\s*</span>", "", html, flags=re.IGNORECASE)

    text = html.strip()
    if text.startswith("<div"):
        matches = list(re.finditer(r"</?div\b[^>]*>", text, flags=re.IGNORECASE))
        open_divs = 0
        end_pos = None

        for m in matches:
            tag = m.group(0).lower()
            if tag.startswith("</div"):
                open_divs -= 1
            else:
                open_divs += 1

            if open_divs == 0:
                end_pos = m.end()
                break

        if end_pos:
            text = text[:end_pos]

    return text.strip()

def is_bad_body_output(html: str) -> bool:
    if not isinstance(html, str):
        return True

    text = html.strip()
    if len(text) < 40:
        return True

    lower = text.lower()
    div_count = lower.count("<div")
    close_div_count = lower.count("</div>")

    keywords = [
        "news and guidance",
        "browse notebooks",
        "track changes",
        "dive deeper",
        "building responsible ai",
        "read our blog",
        "find resources",
        "learn more",
    ]
    text_hits = sum(1 for kw in keywords if kw in lower)

    if div_count > 35 and text_hits == 0:
        return True

    if div_count > close_div_count + 8:
        return True

    repeated_box = 'background-color: #fff; padding: 10px; border-radius: 5px;'
    if lower.count(repeated_box.lower()) >= 8:
        return True

    if div_count >= 25 and "button" not in lower and "img" not in lower and text_hits == 0:
        return True

    return False

def is_layout_color_weak_body_output(html: str) -> bool:
    if not isinstance(html, str):
        return True

    text = html.lower().strip()
    if len(text) < 60:
        return True

    div_count = text.count("<div")
    if div_count < 8:
        return True

    visual_signals = 0
    for token in [
        "background",
        "border",
        "border-radius",
        "display:flex",
        "display:grid",
        "gap:",
        "padding:",
        "margin:",
        "box-shadow",
    ]:
        if token in text:
            visual_signals += 1

    if visual_signals < 4:
        return True

    if text.count("#") < 6:
        return True

    if ("background-color" not in text and "border:" not in text and "border-radius" not in text):
        return True

    return False

def is_shape_color_weak_body_output(html: str) -> bool:
    if not isinstance(html, str):
        return True

    text = html.lower().strip()
    if len(text) < 80:
        return True

    block_visual_hits = 0
    for token in [
        "background-color",
        "border:",
        "border-radius",
        "min-height",
        "height:",
        "width:",
        "display:flex",
        "gap:",
        "padding:",
    ]:
        if token in text:
            block_visual_hits += 1

    if block_visual_hits < 5:
        return True

    if text.count("#") < 8:
        return True

    return False

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form(...),
):
    try:
        print("========== NEW REQUEST ==========")
        print("Mode received:", mode)

        if mode not in PROMPTS:
            return {"error": f"Invalid mode: {mode}"}

        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        image = resize_image_header_aware(image, mode)
        image = resize_image_footer_aware(image, mode)
        image = resize_image_body_aware(image, mode)

        model = get_model(mode)
        prompt = PROMPTS[mode]

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(model.device)

        if mode == "footer":
            ratio = image.height / max(image.width, 1)

            if ratio < 0.18:
                max_new_tokens = 500
            elif ratio < 0.30:
                max_new_tokens = 700
            else:
                max_new_tokens = 900

            repetition_penalty = 1.03

        elif mode == "header":
            max_new_tokens, repetition_penalty = get_header_params(image)

        elif mode == "body":
            max_new_tokens, repetition_penalty = get_body_params(image)

        elif mode == "full":
            aspect = image.height / max(image.width, 1)
            if aspect <= 0.55:
                max_new_tokens = 1400
            elif aspect <= 0.70:
                max_new_tokens = 1200
            else:
                max_new_tokens = 1150

            repetition_penalty = 1.0

        elif mode == "login":
            aspect = image.height / max(image.width, 1)
            if aspect >= 1.9:
                max_new_tokens = 900
            elif aspect >= 0.70:
                max_new_tokens = 580
            else:
                max_new_tokens = 420
            repetition_penalty = 1.10

        elif mode == "regist":
            aspect = image.height / max(image.width, 1)
            if aspect >= 1.0:
                max_new_tokens = 720
            elif aspect >= 0.6:
                max_new_tokens = 680
            else:
                max_new_tokens = 530
            repetition_penalty = 1.10

        elif mode == "forget":
            aspect = image.height / max(image.width, 1)

            if aspect >= 1.00:
                max_new_tokens = 820
            elif aspect >= 0.7:
                max_new_tokens = 720
            elif aspect >= 0.6:
                max_new_tokens = 680
            elif aspect >= 0.4:
                max_new_tokens = 620
            else:
                max_new_tokens = 530

            repetition_penalty = 1.10

        else:
            max_new_tokens = 500
            repetition_penalty = 1.0

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = output[0, input_len:]
        raw_html = processor.decode(generated_tokens, skip_special_tokens=True)

        if mode == "header":
            cleaned_html = clean_html_header(raw_html)
        else:
            cleaned_html = clean_html(raw_html)

        if mode == "login":
            cleaned_html = sanitize_login_html(cleaned_html)

        if mode == "body" and is_bad_body_output(cleaned_html):
            retry_inputs = processor(
                text=BODY_RETRY_PROMPT,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                retry_output = model.generate(
                    **retry_inputs,
                    max_new_tokens=1180,
                    do_sample=False,
                    repetition_penalty=1.02,
                    use_cache=True,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            retry_input_len = retry_inputs["input_ids"].shape[1]
            retry_generated_tokens = retry_output[0, retry_input_len:]
            retry_raw_html = processor.decode(retry_generated_tokens, skip_special_tokens=True)
            retry_cleaned_html = clean_html(retry_raw_html)

            if not is_bad_body_output(retry_cleaned_html):
                raw_html = retry_raw_html
                cleaned_html = retry_cleaned_html

            del retry_inputs, retry_output, retry_generated_tokens
            torch.cuda.empty_cache()
            gc.collect()

        if mode == "body" and (not is_bad_body_output(cleaned_html)) and is_layout_color_weak_body_output(cleaned_html):
            retry2_inputs = processor(
                text=BODY_LAYOUT_COLOR_RETRY_PROMPT,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                retry2_output = model.generate(
                    **retry2_inputs,
                    max_new_tokens=1080,
                    do_sample=False,
                    repetition_penalty=1.03,
                    use_cache=True,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            retry2_input_len = retry2_inputs["input_ids"].shape[1]
            retry2_generated_tokens = retry2_output[0, retry2_input_len:]
            retry2_raw_html = processor.decode(retry2_generated_tokens, skip_special_tokens=True)
            retry2_cleaned_html = clean_html(retry2_raw_html)

            if (not is_bad_body_output(retry2_cleaned_html)) and (not is_layout_color_weak_body_output(retry2_cleaned_html)):
                raw_html = retry2_raw_html
                cleaned_html = retry2_cleaned_html

            del retry2_inputs, retry2_output, retry2_generated_tokens
            torch.cuda.empty_cache()
            gc.collect()

        if mode == "body" and (not is_bad_body_output(cleaned_html)) and is_shape_color_weak_body_output(cleaned_html):
            retry3_inputs = processor(
                text=BODY_SHAPE_COLOR_RETRY_PROMPT,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                retry3_output = model.generate(
                    **retry3_inputs,
                    max_new_tokens=1040,
                    do_sample=False,
                    repetition_penalty=1.03,
                    use_cache=True,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            retry3_input_len = retry3_inputs["input_ids"].shape[1]
            retry3_generated_tokens = retry3_output[0, retry3_input_len:]
            retry3_raw_html = processor.decode(retry3_generated_tokens, skip_special_tokens=True)
            retry3_cleaned_html = clean_html(retry3_raw_html)

            if (not is_bad_body_output(retry3_cleaned_html)) and (not is_shape_color_weak_body_output(retry3_cleaned_html)):
                raw_html = retry3_raw_html
                cleaned_html = retry3_cleaned_html

            del retry3_inputs, retry3_output, retry3_generated_tokens
            torch.cuda.empty_cache()
            gc.collect()

        del inputs, output, generated_tokens
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "mode": mode,
            "code": cleaned_html,
            "raw_code": raw_html,
        }

    except Exception as e:
        import traceback
        print("===== COLAB CRASH =====")
        traceback.print_exc()
        print("=======================")
        return {"error": str(e)}

# =========================
# RUN SERVER
# =========================
def run():
    uvicorn.run(app, host="0.0.0.0", port=)

threading.Thread(target=run).start()
