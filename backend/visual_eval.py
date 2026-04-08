import asyncio
import tempfile
import re
import os
import shutil
from collections import Counter

from playwright.async_api import async_playwright
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

try:
    import pytesseract

    def _find_tesseract_cmd():
        candidates = [
            shutil.which("tesseract"),
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    _tesseract_cmd = _find_tesseract_cmd()
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
        OCR_AVAILABLE = True
    else:
        pytesseract = None
        OCR_AVAILABLE = False
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False


async def render_html_to_image(html):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        path = tmp.name

    wrapped_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          background: white;
        }}
      </style>
    </head>
    <body>
      {html}
    </body>
    </html>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        await page.set_content(wrapped_html)
        await page.screenshot(path=path, full_page=True)

        await browser.close()

    return path


def load_rgb(img_path, size=(512, 512)):
    img = Image.open(img_path).convert("RGB").resize(size)
    return np.array(img)

def load_gray(img_path, size=(512, 512)):
    img = Image.open(img_path).convert("L").resize(size)
    return np.array(img)

# =========================
# TEXT SIMILARITY
# =========================
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def extract_text_ocr(img_path):
    if not OCR_AVAILABLE or pytesseract is None:
        return ""

    try:
        img = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return normalize_text(text)
    except Exception:
        return ""

def tokenize_text(text):
    text = normalize_text(text)
    if not text:
        return []
    return text.split(" ")

def compute_token_f1(target_text, pred_text):
    target_tokens = tokenize_text(target_text)
    pred_tokens = tokenize_text(pred_text)

    if len(target_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(target_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    target_counter = Counter(target_tokens)
    pred_counter = Counter(pred_tokens)

    matched = sum((target_counter & pred_counter).values())

    precision = matched / max(len(pred_tokens), 1)
    recall = matched / max(len(target_tokens), 1)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)

def get_text_region_mask(gray):
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 20 and w >= 3 and h >= 3:
            cleaned[labels == i] = 255

    return cleaned

def compute_text_similarity(img1_path, img2_path):
    gray1 = load_gray(img1_path)
    gray2 = load_gray(img2_path)

    if OCR_AVAILABLE:
        text1 = extract_text_ocr(img1_path)
        text2 = extract_text_ocr(img2_path)

        if text1 and text2:
            score = compute_token_f1(text1, text2)
            return float(score)

        if not text1 and not text2:
            return 1.0

    mask1 = get_text_region_mask(gray1)
    mask2 = get_text_region_mask(gray2)

    score = ssim(mask1, mask2, data_range=255)
    score = max(0.0, min(1.0, score))
    return float(score)

# =========================
# COLOR SIMILARITY
# =========================
def compute_color_similarity(img1_path, img2_path):
    img1 = load_rgb(img1_path)
    img2 = load_rgb(img2_path)

    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)

    mae = np.mean(np.abs(lab1 - lab2))
    score = 1.0 - (mae / 255.0)

    score = max(0.0, min(1.0, score))
    return float(score)

# =========================
# SHAPE SIMILARITY
# =========================
def compute_shape_similarity(img1_path, img2_path):
    gray1 = load_gray(img1_path)
    gray2 = load_gray(img2_path)

    edges1 = cv2.Canny(gray1, 80, 160)
    edges2 = cv2.Canny(gray2, 80, 160)

    score = ssim(edges1, edges2, data_range=255)
    score = max(0.0, min(1.0, score))
    return float(score)

# =========================
# LAYOUT SIMILARITY
# =========================
STRICT_FORM_MODES = {"login", "regist", "forget"}
COMPONENT_LAYOUT_MODES = {"header"}

def get_foreground_mask(gray, threshold=245):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return merged

def crop_to_content(mask, gray, pad=10):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return gray

    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(gray.shape[1], int(xs.max()) + pad + 1)
    y2 = min(gray.shape[0], int(ys.max()) + pad + 1)

    if x2 <= x1 or y2 <= y1:
        return gray

    return gray[y1:y2, x1:x2]

def get_layout_map_full(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    small = cv2.resize(merged, (64, 64), interpolation=cv2.INTER_AREA)
    return small

def get_layout_map_component(gray):
    mask = get_foreground_mask(gray, threshold=245)
    cropped = crop_to_content(mask, gray, pad=10)

    blur = cv2.GaussianBlur(cropped, (5, 5), 0)
    _, thr = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    small = cv2.resize(merged, (64, 64), interpolation=cv2.INTER_AREA)
    return small

def get_layout_map_footer(gray):
    h, w = gray.shape
    start_y = int(h * 0.50)
    bottom_region = gray[start_y:, :]

    mask = get_foreground_mask(bottom_region, threshold=250)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        crop = bottom_region
    else:
        pad = 10
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(bottom_region.shape[1], int(xs.max()) + pad + 1)
        y2 = min(bottom_region.shape[0], int(ys.max()) + pad + 1)

        if x2 <= x1 or y2 <= y1:
            crop = bottom_region
        else:
            crop = bottom_region[y1:y2, x1:x2]

    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, thr = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    small = cv2.resize(merged, (64, 64), interpolation=cv2.INTER_AREA)
    return small

def get_form_detection_region(gray):
    mask = get_foreground_mask(gray, threshold=245)
    cropped = crop_to_content(mask, gray, pad=10)
    return cropped

def detect_form_controls(gray):

    region = get_form_detection_region(gray)

    blur = cv2.GaussianBlur(region, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h, w = region.shape

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < 300:
            continue
        if bw < max(40, int(w * 0.18)):
            continue
        if bh < 12:
            continue
        if bh > int(h * 0.35):
            continue

        aspect = bw / max(bh, 1)
        if aspect < 1.8:
            continue

        boxes.append((x, y, bw, bh))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    merged_boxes = []
    for box in boxes:
        x, y, bw, bh = box
        if not merged_boxes:
            merged_boxes.append(box)
            continue

        px, py, pw, ph = merged_boxes[-1]

        same_row = abs(y - py) < 10
        overlap_x = not (x > px + pw or px > x + bw)

        if same_row and overlap_x:
            nx = min(x, px)
            ny = min(y, py)
            nr = max(x + bw, px + pw)
            nb = max(y + bh, py + ph)
            merged_boxes[-1] = (nx, ny, nr - nx, nb - ny)
        else:
            merged_boxes.append(box)

    return region, merged_boxes

def build_control_layout_map(gray, out_size=(128, 128)):
    region, boxes = detect_form_controls(gray)
    h, w = region.shape

    canvas = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in boxes:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), 255, -1)

    small = cv2.resize(canvas, out_size, interpolation=cv2.INTER_NEAREST)
    return small, boxes

def compute_form_layout_similarity(img1_path, img2_path):
    gray1 = load_gray(img1_path)
    gray2 = load_gray(img2_path)

    map1, boxes1 = build_control_layout_map(gray1, out_size=(128, 128))
    map2, boxes2 = build_control_layout_map(gray2, out_size=(128, 128))

    base_score = ssim(map1, map2, data_range=255)
    base_score = max(0.0, min(1.0, base_score))

    count1 = len(boxes1)
    count2 = len(boxes2)

    if max(count1, count2) == 0:
        count_factor = 1.0
    else:
        count_factor = 1.0 - (abs(count1 - count2) / max(count1, count2))
        count_factor = max(0.0, min(1.0, count_factor))

    proj1 = np.sum(map1 > 0, axis=1).astype(np.float32)
    proj2 = np.sum(map2 > 0, axis=1).astype(np.float32)

    if proj1.max() > 0:
        proj1 = proj1 / proj1.max()
    if proj2.max() > 0:
        proj2 = proj2 / proj2.max()

    projection_score = 1.0 - np.mean(np.abs(proj1 - proj2))
    projection_score = max(0.0, min(1.0, projection_score))

    if count1 != count2:
        final_score = (
            0.45 * min(base_score, 0.90) +
            0.30 * count_factor +
            0.25 * projection_score
        )
    else:
        final_score = (
            0.55 * base_score +
            0.20 * count_factor +
            0.25 * projection_score
        )

    final_score = max(0.0, min(0.92, final_score))

    print("STRICT FORM LAYOUT ACTIVE", flush=True)
    print("controls1:", count1, "controls2:", count2, flush=True)
    print("base_score:", base_score, "count_factor:", count_factor, flush=True)
    print("projection_score:", projection_score, flush=True)
    print("final_score:", final_score, flush=True)

    return float(final_score)

def compute_layout_similarity(img1_path, img2_path, mode=None):
    gray1 = load_gray(img1_path)
    gray2 = load_gray(img2_path)

    if mode == "footer":
        layout1 = get_layout_map_footer(gray1)
        layout2 = get_layout_map_footer(gray2)

        score = ssim(layout1, layout2, data_range=255)
        score = max(0.0, min(1.0, score))
        return float(score)

    if mode in STRICT_FORM_MODES:
        return compute_form_layout_similarity(img1_path, img2_path)

    if mode in COMPONENT_LAYOUT_MODES:
        layout1 = get_layout_map_component(gray1)
        layout2 = get_layout_map_component(gray2)
    else:
        layout1 = get_layout_map_full(gray1)
        layout2 = get_layout_map_full(gray2)

    score = ssim(layout1, layout2, data_range=255)
    score = max(0.0, min(1.0, score))
    return float(score)

# =========================
# FINAL METRICS
# =========================
def compute_visual_metrics(img1_path, img2_path, mode=None):
    text_score = compute_text_similarity(img1_path, img2_path)
    color_score = compute_color_similarity(img1_path, img2_path)
    shape_score = compute_shape_similarity(img1_path, img2_path)
    layout_score = compute_layout_similarity(img1_path, img2_path, mode=mode)

    combined_score = (
        0.30 * text_score +
        0.15 * color_score +
        0.25 * shape_score +
        0.30 * layout_score
    )

    return {
        "text_similarity": float(text_score),
        "color_similarity": float(color_score),
        "shape_similarity": float(shape_score),
        "layout_similarity": float(layout_score),
        "combined_similarity": float(combined_score)
    }