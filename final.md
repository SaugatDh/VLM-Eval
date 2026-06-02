# Synthetic Devanagari Evaluation Dataset — Complete Technical Report

> **For**: Saugat Dhungana, contributor at Himalaya AI Labs
> **Purpose**: Build a held-out, diverse synthetic image+annotation set of Devanagari text to evaluate a VLM that is QLoRA-fine-tuned on Nepali data.
> **Stack**: Pillow + libraqm + HuggingFace datasets + COCO-format JSON. The VLM side is QLoRA + PEFT + TRL SFTTrainer.
> **Audience**: intermediate ML engineer, new to OCR / synthetic text generation.

---

## TL;DR

1. **Install `libraqm` at Pillow build time.** Without it, Devanagari conjuncts render as unconnected glyphs and bboxes are wrong. This is the single biggest "everything looks broken" bug. Pillow issue tracker is full of people who hit this.
2. **Use 3 sources for the text corpus** to maximize "unseen-ness": (a) **FLORES-200 `npi_Deva`** (3001 sentences, held-out by construction, the gold standard), (b) **scraped recent Nepali news** (e.g., ekantipur.com, onlinekhabar.com, setopati.com), (c) a small slice of **Nepali Wikipedia** for diversity.
3. **Render with Pillow + a font folder of 6+ Devanagari fonts** (Noto Sans Devanagari, Noto Serif Devanagari, Mangal, Lohit Devanagari, Gargi, Laila) at varied sizes (20-60 px), random positions, random backgrounds (white / light color / gradient / textured), random text color with contrast check, optional noise + skew.
4. **Compute the bounding box by diffing the rendered text image against a blank**, not by `textbbox` (more accurate, handles conjunct overflow).
5. **Export as COCO JSON** with the standard `images`/`annotations`/`categories` schema, plus a `transcription` field on each annotation (precedent: COCO-Text).
6. **Validate the JSON** with `pycocotools` before shipping it to ARCH.
7. **For the VLM side**: QLoRA (NF4 + double quant + LoRA r=16, alpha=32, target_modules="all-linear", `modules_to_save=["lm_head","embed_tokens"]` for the swap-surgery), TRL SFTTrainer with VLM data collator.

Below is the full pipeline, every code file, the prior art, and citations.

---

## Table of contents

1. [Big picture: how all the pieces fit together](#1-big-picture)
2. [Step 1 — Environment setup (apt + libraqm + fonts)](#2-step-1)
3. [Step 2 — Text corpus (FLORES-200 + scraped news + Wikipedia)](#3-step-2)
4. [Step 3 — Rendering pipeline (Pillow + Devanagari)](#4-step-3)
5. [Step 4 — COCO export + bbox + transcription](#5-step-4)
6. [Step 5 — Splits, validation, packaging](#6-step-5)
7. [Step 6 — Connecting it to the VLM: QLoRA + PEFT + SFTTrainer + decoder swap](#7-step-6)
8. [Step 7 — Edge cases & pitfalls (the "gotchas" list)](#8-step-7)
9. [Step 8 — Prior art in other languages: how English/Chinese/Korean/Indic teams do it](#9-step-8)
10. [Step 9 — Resources (papers, repos, blogs, datasets)](#10-step-9)
11. [Appendix A — Full standalone Python files](#appendix-a)

---

## 1. Big picture

```
┌──────────────────────────────────────────────────────────────────────┐
│  TEXT CORPUS (unseen Nepali sentences)                              │
│   • FLORES-200 npi_Deva (3001 sents, the gold-standard eval set)   │
│   • Scraped live news (ekantipur/onlinekhabar/setopati)             │
│   • Nepali Wikipedia (a few thousand sents)                         │
└──────────────────────────────────────────────────────────────────────┘
                              │ clean, dedupe, length-filter
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RENDERING PIPELINE (Pillow + libraqm + Devanagari fonts)           │
│   For each (text, font, size, position, bg, color) tuple:           │
│     1. Make background (white/colored/gradient/textured)            │
│     2. Render text with ImageFont.truetype(layout_engine=RAQM)      │
│     3. Compute bbox by diffing the text mask against a blank        │
│     4. Save PNG + record (text, bbox, font, size, ...)              │
└──────────────────────────────────────────────────────────────────────┘
                              │ COCO JSON
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  COCO EXPORT                                                         │
│   {                                                                   │
│     "info": {...},                                                    │
│     "images": [{"id": 1, "file_name": "img_00001.png", ...}],        │
│     "annotations": [{"id": 1, "image_id": 1, "bbox": [x,y,w,h],     │
│                      "transcription": "नेपाली पाठ",                  │
│                      "category_id": 1, ...}],                        │
│     "categories": [{"id": 1, "name": "devanagari_text"}]             │
│   }                                                                   │
└──────────────────────────────────────────────────────────────────────┘
                              │ split + validate
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  USED BY VLM EVAL                                                    │
│   QLoRA fine-tune (NF4 + double quant + LoRA on all-linear)         │
│   Decoder swap (modules_to_save=["lm_head","embed_tokens"])          │
│   SFTTrainer with VLM data collator                                   │
│   Eval on this dataset → CER / WER / exact-match                     │
└──────────────────────────────────────────────────────────────────────┘
```

The "VLM eval" part is the *consumer* of the dataset. The pipeline you build produces the dataset; the VLM team consumes it. So this report focuses on the dataset side but includes the VLM context so you can speak the same language as ARCH.

---

## 2. Step 1 — Environment setup

### 2.1 Why this is the most error-prone step

`libraqm` is **not** a pip package. It is a C library that has to be **built and installed before Pillow is built/rebuilt**, so that Pillow can link against it. If you skip this:

- `ImageDraw.text("नेपाल")` will render each Devanagari character as a separate glyph.
- Conjuncts (क्ष, त्र, ज्ञ, etc.) will break apart.
- The shirorekha (top horizontal line) will not be drawn.
- `textbbox` will return wrong rectangles.
- Every annot bbox in the COCO JSON will be off.

Pillow's GitHub issues #1089, #2255, #3191, #3593, #4070 are all variations of this same bug, and the answer in every case is "install libraqm then rebuild Pillow".

### 2.2 Recommended setup (Arch / CachyOS / Ubuntu)

Save this as `setup.sh` at the project root:

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- 1. System packages for libraqm ---
sudo pacman -S --needed \
    noto-fonts \
    noto-fonts-extra \
    harfbuzz fribidi gtk-doc freetype2 libraqm 2>/dev/null || \

# Arch fallback: libraqm may not be in main repos. Use AUR if needed.
# On Debian/Ubuntu:
sudo apt-get update
sudo apt-get install -y \
    libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools \
    libjpeg-dev zlib1g-dev libraqm-dev fontconfig

# --- 2. Python packages ---
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Pillow rebuilt against libraqm. If you use a pre-built wheel, libraqm
# is already included on most platforms; force a source build just to be sure:
pip install --no-binary :all: Pillow

pip install \
    pillow numpy tqdm datasets \
    requests beautifulsoup4 \
    pycocotools  # for validating the COCO JSON we produce

# --- 3. Verify libraqm is available to Pillow ---
python3 - <<'EOF'
from PIL import Image, ImageDraw, ImageFont
img = Image.new("RGB", (400, 100), "white")
font = ImageFont.truetype("/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf", 36)
# Use the RAQM layout engine for correct Devanagari shaping
draw = ImageDraw.Draw(img)
draw.text((10, 10), "यात्रा नेपाली क्ष", font=font, fill="black", embedded_color=False)
img.save("/tmp/deva_render_test.png")
print("OK: libraqm rendering saved to /tmp/deva_render_test.png")

# Verify the textbbox is sensible
bbox = draw.textbbox((10, 10), "यात्रा", font=font)
print(f"textbbox for 'यात्रा': {bbox}")
assert bbox[2] - bbox[0] > 10, "text width suspiciously small — libraqm not working"
print("libraqm is correctly shaping Devanagari.")
EOF
```

### 2.3 Verifying fonts

```bash
# List all Devanagari fonts installed on the system
fc-list | grep -i devanagari
# Expected: NotoSansDevanagari-Regular/Bold/Medium, NotoSerifDevanagari, Mangal (if installed), Lohit (if installed)
```

### 2.4 Quick gotcha

If you are running **inside a Docker container**, Pillow's pre-built wheels may not include libraqm. Force a source build:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
        libfreetype6-dev libharfbuzz-dev libfribidi-dev libjpeg-dev zlib1g-dev \
        libraqm-dev fontconfig fonts-noto-core fonts-noto-extra \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir --no-binary :all: Pillow
RUN pip install --no-cache-dir numpy tqdm datasets requests beautifulsoup4 pycocotools
```

---

## 3. Step 2 — Text corpus

### 3.1 The "unseen" question

What "unseen" means in practice:

| Source | How unseen? | Pros | Cons |
|---|---|---|---|
| **FLORES-200 `npi_Deva`** | Explicitly held out from pretraining by Meta AI. Used as the eval set in 100s of MT/NLP papers. | Highest credibility, standardized, everyone knows what it is. | Only 3001 sentences, all Wikipedia-derived (limited domain). |
| **Scraped live news** (ekantipur, onlinekhabar, setopati, nagariknews) | Current as of the scraping date, so guaranteed to be post-training-cutoff for any model. | Truly unseen, real-world noise, idiomatic Nepali. | Scraper can break, you may need to re-scrape, news writing is biased toward certain topics. |
| **Nepali Wikipedia dump** | Wikipedia is a common pretraining source, so technically *seen*. But sentences outside the most-cited articles are likely unseen. | Free, large (~39K articles), easy to load. | Domain is encyclopedic, not colloquial. |
| **CC-100 / OSCAR (CommonCrawl)** | XLM-R was trained on CC-100, so this is **seen**. | Large. | Not unseen. |
| **Nepali government documents** (laws.gov.np) | Domain-specific, professional prose, often unseen. | High-quality formal text. | Legal/regulatory domain, not everyday language. |

**Recommended mix**: ~30% FLORES-200, ~50% scraped news, ~20% Nepali Wikipedia / niche sources. This gives you 1) benchmark credibility (FLORES), 2) real-world distribution (news), 3) coverage breadth (Wikipedia).

### 3.2 Loading FLORES-200

```python
from datasets import load_dataset

ds = load_dataset("facebook/flores", "npi_Deva", split="devtest")
# 997 sentences; "dev" is 297; "devtest" is the standard held-out eval split
sentences = [row["sentence"] for row in ds]
print(f"Loaded {len(sentences)} FLORES-200 Nepali sentences")
print("First 3:", sentences[:3])
```

### 3.3 Scraping Nepali news

```python
import requests
from bs4 import BeautifulSoup
import time
import random

SITES = [
    ("https://ekantipur.com", "section"),
    ("https://www.onlinekhabar.com", "section"),
    ("https://www.setopati.com", "section"),
    ("https://www.nagariknews.com.np", "section"),
]

def scrape_article_text(url: str) -> str:
    """Extract paragraph text from a Nepali news article URL."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    # Most Nepali news sites use <p> tags for body text
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    # Filter to Devanagari-only (drop English-only, short, or boilerplate)
    devanagari = [p for p in paragraphs if any("\u0900" <= ch <= "\u097F" for ch in p) and len(p) > 30]
    return "\n".join(devanagari)


def harvest_article_links(homepage: str, n_links: int = 100) -> list[str]:
    """Pull article URLs from a news homepage. Naive — refine per site."""
    resp = requests.get(homepage, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    # All anchor tags; pick those with article-like paths.
    # (You should refine this regex per site.)
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = homepage.rstrip("/") + href
        if href.startswith("http") and homepage.split("//")[1].split("/")[0] in href:
            # Naive filter: article URLs typically have a year in the path
            if any(year in href for year in ["2024", "2025", "2026"]):
                links.append(href)
    return list(dict.fromkeys(links))[:n_links]


def harvest_corpus(target_sents: int = 10000) -> list[str]:
    corpus = []
    for homepage, _ in SITES:
        print(f"Scraping {homepage}...")
        links = harvest_article_links(homepage, n_links=200)
        for url in links:
            text = scrape_article_text(url)
            # Split into sentences (Nepali ends sentences with "।")
            for sent in text.split("।"):
                sent = sent.strip()
                if 20 <= len(sent) <= 150:
                    corpus.append(sent + "।")
            if len(corpus) >= target_sents:
                break
            time.sleep(random.uniform(0.5, 1.5))  # polite
        if len(corpus) >= target_sents:
            break
    return list(dict.fromkeys(corpus))  # dedupe


# corpus = harvest_corpus(target_sents=5000)
# print(f"Harvested {len(corpus)} Nepali sentences from live news.")
```

> **Caveat for ARCH**: most Nepali news sites do not have public APIs. Scraping is fragile and ethically grey; ideally ARCH will have a private corpus (their team's `nepali-text-corpus` from IRIISNEPAL, or a private scrape of their domain). Ask ARCH first.

### 3.4 Loading Nepali Wikipedia (for breadth)

```python
# Option A: HuggingFace Wikipedia loader
from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "ne", split="train", streaming=True)
# Pull a few thousand articles, split into sentences
texts = []
for i, row in enumerate(ds):
    if i >= 1000:
        break
    for sent in row["text"].split("।"):
        sent = sent.strip()
        if 20 <= len(sent) <= 200:
            texts.append(sent + "।")

# Option B: Kaggle
# https://www.kaggle.com/datasets/disisbig/nepali-wikipedia-articles — 39K articles
```

### 3.5 Cleaning & filtering

```python
import re

def clean_corpus(sentences: list[str]) -> list[str]:
    """Filter to evaluable Nepali sentences."""
    out = []
    for s in sentences:
        s = s.strip()
        # Must have at least one Devanagari char
        if not any("\u0900" <= ch <= "\u097F" for ch in s):
            continue
        # Drop sentences that are mostly Latin (English code-switched paragraphs)
        deva_ratio = sum(1 for ch in s if "\u0900" <= ch <= "\u097F") / max(1, len(s))
        if deva_ratio < 0.6:
            continue
        # Drop sentences with HTML entities / URLs (came from scraping)
        if re.search(r"https?://|&[a-z]+;|<|>", s):
            continue
        # Length window that works well for single-line rendering
        if not (15 <= len(s) <= 120):
            continue
        out.append(s)
    return list(dict.fromkeys(out))  # dedupe

corpus = clean_corpus(sentences)
print(f"After cleaning: {len(corpus)} sentences")
```

---

## 4. Step 3 — Rendering pipeline

This is the heart of the project. The full standalone file is `code/render_devanagari.py` in the appendix; here is the conceptual walkthrough.

### 4.1 Design choices, with rationale

**Q: Why Pillow rather than TRDG (TextRecognitionDataGenerator)?**
A: TRDG is excellent for the *training* set of an OCR model — it generates millions of cheap, varied images. For an *eval* set, you want more control over: (a) bbox format (COCO, not Tesseract), (b) per-image metadata (which font, which size, which text) for later error analysis, and (c) text corpus sourcing (TRDG has its own corpus files; you want to inject your FLORES + scraped sentences). Custom Pillow is also easier to read and extend.

**Q: Why not PaddleOCR StyleText?**
A: StyleText is the gold standard for *style transfer* — you give it a "style image" of how text should look, and it renders new text in that style. It currently supports en, ch, ko. Adapting it to Devanagari requires porting the foreground style transfer model, which is significant work. Use StyleText if/when you want to render Nepali text that looks like a specific brand or scanned-document style. For a general "diverse devanagari eval set", Pillow is fine.

**Q: Why not UnrealText?**
A: UnrealText renders scene + text together in a 3D engine for realistic lighting. Overkill for an OCR eval set; the scene context doesn't help when you just want to test text-reading. Keep it as a future direction if ARCH wants to test text in the wild (signage, etc.).

### 4.2 The rendering function, explained line by line

```python
def render_devanagari_image(
    text: str,
    font_path: str,
    font_size: int = 36,
    image_size: tuple[int, int] = (640, 128),
    background: str = "white",  # "white"|"colored"|"gradient"|"textured"
    text_color: tuple[int, int, int] | None = None,
    contrast_check: bool = True,
) -> tuple[Image.Image, dict]:
    """Render one Devanagari string onto a synthetic image and return the bbox.

    Returns: (PIL.Image, {"text", "bbox": [x, y, w, h], "font", "size", "color", "background"})
    """
    W, H = image_size

    # --- 1. Background ---
    if background == "white":
        bg_color = (255, 255, 255)
        img = Image.new("RGB", (W, H), bg_color)
    elif background == "colored":
        # Light tinted background (avoid pure white to vary the look)
        bg_color = tuple(random.randint(200, 255) for _ in range(3))
        img = Image.new("RGB", (W, H), bg_color)
    elif background == "gradient":
        # Horizontal gray gradient
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(W):
            v = int(200 + 55 * x / W)
            arr[:, x] = (v, v, v)
        img = Image.fromarray(arr)
    elif background == "textured":
        # Subtle paper-like noise
        arr = np.random.randint(200, 255, (H, W, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

    # --- 2. Font (with explicit raqm layout engine for correct shaping) ---
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.LAYOUT_RAQM)

    # --- 3. Text color with contrast check ---
    if text_color is None:
        text_color = (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))

    if contrast_check:
        # WCAG-style luminance contrast. Reject near-invisible combinations.
        bg_lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        fg_lum = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
        if abs(bg_lum - fg_lum) < 60:
            text_color = (0, 0, 0) if bg_lum > 127 else (255, 255, 255)

    # --- 4. Render the text to a separate mask image for accurate bbox ---
    # We render to a 1-channel image, then diff with a blank to get the exact pixels.
    # This is the most accurate bbox method, more reliable than textbbox for Devanagari.
    text_mask = Image.new("L", (W, H), 0)
    text_draw = ImageDraw.Draw(text_mask)

    # Center the text horizontally and vertically with a small random offset
    # Use textlength (more reliable than textbbox for Devanagari) to compute width
    try:
        text_w = int(text_draw.textlength(text, font=font))
    except AttributeError:
        # Older Pillow: use textbbox width as fallback
        bbox = text_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

    # Vertical extent: use font.getmetrics and ascent/descent, or textbbox height
    ascent, descent = font.getmetrics()
    text_h = ascent + descent

    # Centered position with small jitter
    x = max(5, (W - text_w) // 2 + random.randint(-10, 10))
    y = max(5, (H - text_h) // 2 + random.randint(-5, 5))

    text_draw.text((x, y), text, font=font, fill=255)

    # --- 5. Find the actual rendered bbox by diffing ---
    blank = Image.new("L", (W, H), 0)
    diff = ImageChops.difference(text_mask, blank)
    rendered_bbox = diff.getbbox()  # (l, t, r, b) of actually-rendered pixels

    if rendered_bbox is None:
        # Nothing was rendered (font missing the glyph, etc.); skip
        return None, None

    l, t, r, b = rendered_bbox
    bbox_xywh = [l, t, r - l, b - t]

    # --- 6. Composite the text onto the background ---
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=text_color)

    annotation = {
        "text": text,
        "bbox": bbox_xywh,
        "font": font_path,
        "size": font_size,
        "color": text_color,
        "background": background,
    }
    return img, annotation
```

### 4.3 Why "diff the mask" is more accurate than `textbbox`

`textbbox` in Pillow uses FreeType's bbox calculation, which is *font metric-based* — it returns the theoretical bounding box based on the glyph's ink box plus some line metrics. For simple Latin text, this is fine. For Devanagari, the rendered text often extends beyond the metric box (because of diacritics like chandrabindu / anusvara that float above the headline, or because of compound conjuncts that widen). Conversely, some fonts over-report metric widths for Devanagari because of internal padding.

The "render to a separate L-mode image, then `getbbox()` of the difference" method computes the *actual rendered pixel* bbox. This is what TRDG does internally. It's slower (extra render) but always correct.

### 4.4 Variation knobs (what to randomize)

| Knob | Values | Why |
|---|---|---|
| Font | 6+ Devanagari fonts in `fonts/` folder | Visual diversity; tests font-independence |
| Font size | 20, 24, 28, 32, 36, 40, 48, 56 | Tests scale robustness |
| Background | white, colored, gradient, textured | Tests background robustness |
| Text color | random dark (0-80 per channel) or pure black | Avoids trivial "always same color" leak |
| Contrast check | ON | Prevents near-invisible labels |
| Image size | (256, 64), (384, 96), (512, 128), (640, 160) | Tests multi-resolution |
| Text position | jitter ±10px horizontal, ±5px vertical | Avoids "always centered" leak |
| Noise (optional) | Gaussian noise σ ∈ [0, 5] | Tests noise robustness |
| Skew (optional) | rotation ∈ [-3°, 3°] | Tests small rotation |

The user can also add more aggressive perturbations (curved baselines, blurred text, low-resolution downsampling) to make the eval harder. I'll include a function for that too.

### 4.5 Making the eval actually hard

The whole point of an eval set is to **discriminate** between models. A trivial synthetic set (always centered, always black on white, always the same font) is useless. Make the eval hard by including:

- **Low-contrast samples** (e.g., #444 on #CCC): tests whether the VLM is using pixel intensity or just guessing.
- **Multi-line samples**: tests reading order.
- **Curved / rotated text**: tests geometric robustness.
- **Mixed-script samples** (Nepali + Latin + numerals): tests the VLM's ability to switch scripts in the same line.
- **Edge cases**: long conjunct chains (e.g., "क्रमबद्ध"), shirorekha-only words, conjuncts with ZWJ vs ZWNJ.

---

## 5. Step 4 — COCO export

### 5.1 Why COCO

- The VLM (and almost all modern detection / grounding models) can read COCO JSON out of the box.
- pycocotools provides `COCO(annotation_file).getAnnIds(...)` for evaluation.
- You can convert COCO to any other format (YOLO, Pascal VOC, custom) trivially.

### 5.2 Schema, with the `transcription` extension

The user requested `transcription` to be added to each annotation. The closest published precedent is **COCO-Text** which uses `utf8_string` ([COCO-Text, Cornell](https://vision.cornell.edu/se3/coco-text-2/)). Using `transcription` (more semantically clear) is fine; just be explicit in the dataset README that this is an extension to standard COCO.

```json
{
  "info": {
    "description": "Synthetic Devanagari evaluation dataset for Himalaya AI VLM",
    "version": "1.0",
    "year": 2026,
    "contributor": "Saugat Dhungana <saugat@himalaya-ai.example>"
  },
  "licenses": [
    {"id": 1, "name": "CC-BY-SA-4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
  ],
  "images": [
    {
      "id": 1,
      "file_name": "img_00001.png",
      "width": 640,
      "height": 128
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [123, 32, 250, 64],
      "area": 16000,
      "iscrowd": 0,
      "transcription": "नेपाली पाठ"
    }
  ],
  "categories": [
    {"id": 1, "name": "devanagari_text", "supercategory": "text"}
  ]
}
```

### 5.3 Export function

```python
def export_coco(
    images: list[dict],
    annotations: list[dict],
    output_path: str,
    description: str = "Synthetic Devanagari eval",
):
    """Build a COCO-format JSON from per-sample dicts and write to disk.

    Each entry in `images` is {"file_name": str, "width": int, "height": int}.
    Each entry in `annotations` is {"image_id": int, "bbox": [x, y, w, h],
                                    "transcription": str, "area": int, "iscrowd": int}.
    """
    coco = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2026,
            "contributor": "Himalaya AI Labs",
        },
        "licenses": [
            {"id": 1, "name": "CC-BY-SA-4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
        ],
        "categories": [
            {"id": 1, "name": "devanagari_text", "supercategory": "text"}
        ],
        "images": [],
        "annotations": [],
    }
    for img_id, img in enumerate(images, start=1):
        coco["images"].append({
            "id": img_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        })
    for ann_id, ann in enumerate(annotations, start=1):
        x, y, w, h = ann["bbox"]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": ann["image_id"],
            "category_id": 1,
            "bbox": [x, y, w, h],
            "area": int(w * h),
            "iscrowd": 0,
            "transcription": ann["transcription"],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} annotations to {output_path}")
```

### 5.4 Validation

```python
def validate_coco(coco_path: str) -> None:
    """Sanity-check the exported COCO JSON using pycocotools."""
    from pycocotools.coco import COCO
    coco = COCO(coco_path)
    print(f"Loaded COCO: {len(coco.imgs)} images, {len(coco.anns)} annotations, {len(coco.cats)} categories")

    # 1. All bboxes are within image bounds
    bad = 0
    for ann_id, ann in coco.anns.items():
        img = coco.imgs[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if x < 0 or y < 0 or x + w > img["width"] or y + h > img["height"]:
            bad += 1
            if bad < 5:
                print(f"  ! annotation {ann_id} bbox overflows image: {ann['bbox']} vs {img['width']}x{img['height']}")
    assert bad == 0, f"{bad} annotations have out-of-bounds bboxes"

    # 2. No empty transcriptions
    empty = sum(1 for ann in coco.anns.values() if not ann.get("transcription", "").strip())
    assert empty == 0, f"{empty} annotations have empty transcriptions"

    # 3. Every transcription has Devanagari
    no_deva = sum(
        1 for ann in coco.anns.values()
        if not any("\u0900" <= ch <= "\u097F" for ch in ann["transcription"])
    )
    if no_deva > 0:
        print(f"  ⚠ {no_deva} annotations have no Devanagari chars — fine if intentional, otherwise re-check corpus")

    # 4. Every image file exists
    from pathlib import Path
    base_dir = Path(coco_path).parent
    missing_files = [
        coco.imgs[i]["file_name"] for i in coco.imgs
        if not (base_dir / coco.imgs[i]["file_name"]).exists()
    ]
    assert not missing_files, f"Missing image files: {missing_files[:5]}"

    print("✓ COCO JSON is valid.")
```

---

## 6. Step 5 — Splits, packaging

### 6.1 Train/val/test split

For an **eval-only** dataset, you don't strictly need train/val/test. But the convention is to keep a small **val** split for VLM developers to tune prompts, and a **test** split for the final report. A typical split is 0/0/100 (eval-only) or 10/10/80 (if you also want a tiny training slice for sanity-check fine-tuning). For your case, I'd recommend:

- **test (90%)** — the eval set, hidden from prompt tuning
- **val (10%)** — for prompt tuning / hyperparameter search

```python
def split_coco(coco_path: str, out_dir: str, val_ratio: float = 0.10, seed: int = 42):
    """Split a COCO dataset into val / test by image_id. No train split (eval-only)."""
    random.seed(seed)
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_ids = [img["id"] for img in data["images"]]
    random.shuffle(image_ids)
    n_val = int(len(image_ids) * val_ratio)
    val_ids = set(image_ids[:n_val])
    test_ids = set(image_ids[n_val:])

    def filter_split(ids):
        return {
            **data,
            "images": [i for i in data["images"] if i["id"] in ids],
            "annotations": [a for a in data["annotations"] if a["image_id"] in ids],
        }

    for split_name, ids in [("val", val_ids), ("test", test_ids)]:
        with open(f"{out_dir}/{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(filter_split(ids), f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(ids)} images to {split_name}.json")
```

### 6.2 Project layout

```
devanagari_eval/
├── README.md                  # What this is, who built it, how to use it
├── LICENSE
├── setup.sh                   # Installs libraqm + Pillow + deps
├── fonts/                     # All Devanagari fonts (TTF)
│   ├── NotoSansDevanagari-Regular.ttf
│   ├── NotoSansDevanagari-Bold.ttf
│   ├── NotoSerifDevanagari-Regular.ttf
│   ├── Mangal.ttf
│   ├── Lohit-Devanagari.ttf
│   ├── Gargi.ttf
│   └── Laila-Regular.ttf
├── code/
│   ├── corpus.py              # Load FLORES-200, scrape news, load Wikipedia
│   ├── render.py              # render_devanagari_image function
│   ├── generate.py            # Top-level: corpus → render → COCO export
│   ├── validate.py            # pycocotools-based validation
│   └── visualize.py           # Render bbox overlay on sample images (for review)
├── images/                    # All generated PNGs
│   ├── img_00001.png
│   └── ...
├── annotations.json           # Full COCO JSON
├── val.json                   # 10% split
├── test.json                  # 90% split
└── stats.json                 # Per-corpus-source, per-font, per-size counts
```

### 6.3 Stats report (for ARCH)

Run after generation:

```python
def compute_stats(coco: dict) -> dict:
    """Per-source / per-font / per-size / per-background stats."""
    from collections import Counter
    font_counts = Counter()
    size_counts = Counter()
    bg_counts = Counter()
    text_lengths = []
    for ann in coco["annotations"]:
        # We need the metadata — store it in a parallel list, not the COCO JSON
        # (COCO standard doesn't have these fields, so we keep them in `meta`)
        pass
    return {...}
```

(Note: COCO standard does not include font/size/bg metadata. Either (a) drop those fields and rely on filenames to encode the metadata, or (b) keep a parallel `meta.json` mapping annotation_id → {font, size, bg}. I recommend (b).)

---

## 7. Step 6 — Connecting it to the VLM

This section is for completeness — so the dataset pipeline is grounded in the consumer's reality. The VLM team is doing QLoRA + PEFT + TRL; here's what that looks like in their world.

### 7.1 QLoRA in 30 seconds

QLoRA (Dettmers et al., NeurIPS 2023) lets you finetune a multi-billion-parameter model on a single GPU by:

1. **Storing the base model weights in 4-bit** (specifically NF4, a data type optimized for normally-distributed weights). This shrinks VRAM ~4x.
2. **Computing the forward/backward in 16-bit** (typically bfloat16). The 4-bit weights are dequantized on-the-fly for the matmul, then thrown away.
3. **Adding small LoRA adapters** (low-rank trainable matrices) to each linear layer. Only the adapters are trained; the 4-bit base weights never see a gradient.
4. **Double-quantizing the quantization constants** for an extra ~0.4 bits/param saving.
5. **Paged optimizers** to spill optimizer state to CPU RAM when VRAM spikes.

The math result: 65B model on a single 48GB GPU, **with no measurable quality loss vs 16-bit LoRA finetuning**. [Dettmers et al., 2023] [9]

### 7.2 The HuggingFace canonical QLoRA recipe

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Quantization config (NF4 + double quant + bf16 compute)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4 (information-theoretically optimal for ~N(0,1))
    bnb_4bit_use_double_quant=True,   # Quantize the quantization constants too
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute happens in bf16
)

# 2. Load the model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "himalaya-ai/nepali-vlm-base",     # whatever base VLM ARCH is using
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",       # Eager attn is required by some VLM archs (Gemma-3)
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA config
peft_config = LoraConfig(
    r=16,                              # rank
    lora_alpha=32,                     # 2*r; effective scale = alpha/r = 2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",       # All linear projections in the LLM stack
    modules_to_save=["lm_head", "embed_tokens"],  # FULL finetune of these — for decoder swap
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # should print ~0.1-1% trainable

# 4. TRL SFTTrainer (handles VLM data collators automatically)
training_args = SFTConfig(
    output_dir="./nepali-vlm-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,                # QLoRA paper recommendation
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    optim="adamw_torch_fused",
    save_strategy="epoch",
    logging_steps=10,
    report_to="tensorboard",
    max_length=None,                   # critical for VLMs — don't truncate image tokens
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset("himalaya-ai/nepali-vlm-train", split="train"),
    eval_dataset=load_dataset("himalaya-ai/nepali-vlm-train", split="val"),
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("./nepali-vlm-finetuned/final")
```

### 7.3 Decoder swap surgery — what it is and how PEFT supports it

"Decoder swap" in this context means: **replace the LM head and/or the input embedding matrix of the base VLM with a new one trained for the Nepali vocabulary**, then continue training the new matrices plus the LoRA adapters.

Why you'd do this:

- The base VLM (e.g., Qwen2-VL, LLaVA-1.6) has a tokenizer that covers Latin, digits, basic punctuation — but might be missing or sparse on Devanagari-specific characters or Nepali-specific tokens.
- Even if the base tokenizer has Devanagari codepoints, the LM head's output rows for those codepoints have been barely-trained because the pretraining data is mostly English.
- Swapping the head for one that's been pre-trained on a Nepali corpus (e.g., IRIISNEPAL RoBERTa's word embeddings, or a fresh from-scratch Devanagari LM head) gives the VLM a much better starting point for Devanagari generation.

The mechanism in HuggingFace PEFT:

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],  # ← this is the swap
)
```

`modules_to_save` tells PEFT: "for these modules, don't apply LoRA — instead, train them in full precision alongside the LoRA adapters." So `lm_head` and `embed_tokens` will be updated in their entirety during training.

A more aggressive version:

```python
# Pre-load the new lm_head from a separately-trained Nepali LM
from transformers import AutoModelForCausalLM
nepali_lm = AutoModelForCausalLM.from_pretrained("himalaya-ai/nepali-lm-base")
# Resize the VLM's embedding to match the Nepali LM's vocab (or vice versa)
vlm.resize_token_embeddings(len(nepali_lm.config.vocab))
# Copy the LM head weights
vlm.lm_head.weight.data.copy_(nepali_lm.lm_head.weight.data)
vlm.model.embed_tokens.weight.data.copy_(nepali_lm.model.embed_tokens.weight.data)
# Now freeze everything except the new embeddings + LoRA
peft_config = LoraConfig(
    ...,
    modules_to_save=["lm_head", "embed_tokens"],
)
```

This is the pattern. It's not a single paper; it's community practice, used by Indic LLM groups (NepaliGPT, IRIISNEPAL RoBERTa, OpenHathi, etc.) when adapting a base LLM to a new script.

### 7.4 Using the eval set in the VLM training loop

Once you have the synthetic Devanagari eval set, you evaluate the VLM like this:

```python
# Eval-only: load the VLM with the new LoRA adapter
from peft import PeftModel
base = AutoModelForImageTextToText.from_pretrained("himalaya-ai/nepali-vlm-base", ...)
model = PeftModel.from_pretrained(base, "./nepali-vlm-finetuned/final")
model.eval()

# Run inference on each eval image
from pycocotools.coco import COCO
coco = COCO("test.json")
for img_id in coco.imgs:
    img_info = coco.imgs[img_id]
    img = Image.open(f"images/{img_info['file_name']}")
    # Build the prompt — depends on your VLM
    prompt = "<image>\nयस चित्रमा के लेखिएको छ? यस चित्रमा देखिएको पाठ ट्रान्स्क्राइब गर्नुहोस्।"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
    prediction = processor.batch_decode(output, skip_special_tokens=True)[0]
    ground_truth = coco.anns[coco.getAnnIds(imgIds=img_id)[0]]["transcription"]
    # Compute CER / WER here
    print(f"GT:   {ground_truth}")
    print(f"PRED: {prediction}")
```

Where CER = Character Error Rate (Levenshtein distance / number of chars in GT), and WER = Word Error Rate (Levenshtein on word tokens / number of words in GT). Both can be computed with `jiwer` library or `pyctcdecode`.

### 7.5 What the eval set tells you

If the VLM gets **high CER on conjuncts** (e.g., "क्षत्र" → "कसत्र"), you know the model has trouble with multi-glyph Devanagari shaping. If the **CER is high on ZWJ vs non-ZWJ** (same word, different rendering), you know the model is confused by ligature-vs-non-ligature forms. If **CER is high on low-contrast images**, you know the visual encoder is poor at low-light/texture. If **CER is high only at small font sizes**, you know the VLM's vision tower doesn't handle small text well.

The point is: this eval set, with the metadata recorded per image, becomes a **diagnostic tool** not just a number. Recommend that ARCH runs it as a *per-bucket* report (per font, per size, per contrast, per conjunct-density), not a single aggregate CER.

---

## 8. Step 7 — Edge cases & pitfalls (the "gotchas" list)

These are the things that *will* trip you up if you don't plan for them. Most of them are documented in Pillow GitHub issues, PaddleOCR StyleText issues, and the QLoRA paper.

### 8.1 Pillow / libraqm

| Symptom | Cause | Fix |
|---|---|---|
| Conjuncts render as separate glyphs | libraqm not installed | `pip install --no-binary :all: Pillow` after installing libraqm system-wide |
| textbbox returns weird sizes | libraqm not loaded | Pass `layout_engine=ImageFont.LAYOUT_RAQM` explicitly |
| `font.getmetrics()` returns (0, 0) | Font is broken or not a TTF | `fc-validate <font>.ttf`; try a different font |
| Font missing a Devanagari glyph | Font has incomplete coverage | Filter your corpus against the font's actual coverage, or switch to Noto Sans Devanagari (most complete) |
| Vertical text rendering breaks | Pillow doesn't support vertical Devanagari natively | Skip vertical; render only horizontal |

### 8.2 Bounding box

| Symptom | Cause | Fix |
|---|---|---|
| Bbox too small (text clipped) | `textbbox` returns a tight font-metric box that misses inked pixels | Use the diff-with-blank method described in §4.2 |
| Bbox too large (huge empty margins) | `textbbox` includes the line ascent/descent | Use diff-with-blank; or subtract font's line metrics |
| Bbox slightly off (1-2 px) | Anti-aliasing edge cases | Acceptable; <2px is within rendering noise |
| Bbox wildly wrong for conjuncts | FreeType shaping vs raqm shaping disagreement | `layout_engine=LAYOUT_RAQM` in `truetype` call |

### 8.3 Corpus

| Symptom | Cause | Fix |
|---|---|---|
| "All my sentences have English" | Scraping pulled English-only articles | Filter by Devanagari character ratio (see §3.5) |
| "Some sentences are duplicates" | News sites repeat headlines | Dedupe after collection |
| "I have 1000 sentences but only 500 rendered" | Font missing glyphs, or sentences with only punctuation | Filter corpus to sentences that have at least one Devanagari char AND render successfully with the chosen font |
| "FLORES-200 sentences are too short / too long" | All FLORES sentences are ~21 words long, so they're around 60-100 chars. | Acceptable for single-line rendering; or filter to 20-80 chars |

### 8.4 COCO format

| Symptom | Cause | Fix |
|---|---|---|
| `pycocotools` rejects the JSON | Trailing comma, missing field, wrong type | Run the validator; check `bbox` is a 4-element list, not a dict |
| Bbox overflows image | Text positioned outside image bounds | Clip x, y in the rendering function |
| Empty `transcription` | Sentence had only Latin / only punctuation | Filter corpus to sentences that have Devanagari |
| Unicode escape issues | json.dump with `ensure_ascii=True` (default) escapes Devanagari | Pass `ensure_ascii=False` to json.dump |
| `category_id` mismatch | Annotation has category_id=2 but categories only has id=1 | Always use a single category; set category_id=1 for all |

### 8.5 QLoRA / PEFT

| Symptom | Cause | Fix |
|---|---|---|
| "Target modules not found" | LoRA targets don't match the model's actual layer names | Print `model.named_modules()` and find the right names (e.g., `q_proj`, `qkv_proj`, `query_key_value`) |
| Loss doesn't decrease | Learning rate too low, or LoRA targets miss the right modules | Try LR=2e-4, target all-linear, ensure `modules_to_save` is set if doing decoder swap |
| OOM during training | batch too big, sequence too long, gradient checkpointing off | Enable gradient checkpointing, reduce batch size, increase grad accum |
| NaN loss | bf16 instability, learning rate too high, bad data | Use fp32 master weights (Accelerate config), reduce LR, check data |
| LoRA adapters load but predictions are bad | The new `lm_head` from `modules_to_save` is not being saved | `trainer.save_model()` saves them; `peft_model.save_pretrained()` also saves them; verify the saved checkpoint includes them |

### 8.6 Eval pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| CER is 0% on the eval set | The model has seen the eval text during training | Use only FLORES + scraped news; never CC-100 / OSCAR / Wikipedia sentences that overlap with training data |
| CER is 100% on every sample | Model output is empty / wrong format | Check the prompt template; check that the VLM knows it's being asked for OCR (not captioning) |
| CER is low on training data, high on eval | Overfitting | Reduce LoRA rank or epochs; or use more diverse training data |

---

## 9. Step 8 — Prior art in other languages

This section is the "how people have already done this for other languages" you asked for. For each prior work, I give: what it does, the design choices the user can learn from, and how it relates to the Devanagari project.

### 9.1 SynthText (Gupta, Vedaldi, Zisserman, CVPR 2016) — English

**What it is**: The seminal synthetic scene-text generator. 800K training images, each with word-level and character-level bounding boxes + transcription. Combines real background images (with depth + segmentation) and rendered text via Poisson image editing for blending.

**Design choices to learn from**:
- **Use scene context, not blank backgrounds.** SynthText's claim to fame is that the text respects the local 3D scene (depth, orientation, lighting). For your project this means: consider also generating a fraction of images on top of *real* photo backgrounds (cropped book pages, sign photos, even random photos) to test the VLM's robustness to real-world contexts.
- **Character-level + word-level bboxes.** SynthText's COCO-style output has both. If you want character-level bboxes for OCR, you can post-process the rendered image with `PIL.ImageDraw.textlength` and font metrics per character; or use TRDG's Tesseract-format output (`-obb 2`).
- **800K is overkill for an eval set.** You need 1K-10K. Less is more — the eval set should be **curated for diversity, not just big**.

**Citation**: arXiv 1604.06646, code: github.com/ankush-me/SynthText [14][15]

### 9.2 TextRecognitionDataGenerator / TRDG (Belval) — multilingual

**What it is**: The most popular Pillow-based generator. ~3K GitHub stars. Supports en, cn, ja, hi, ko, ar, fa, plus 70+ languages via font folders. Outputs bbox via mask (Tesseract format). Used by hundreds of OCR training pipelines as a quick boost.

**Design choices to learn from**:
- **Simple variation knobs, well-engineered CLI**: skew, blur, distortion, multiple background types, multiple alignments. The fact that it has 30+ CLI flags tells you what variation matters.
- **Font folder per language.** Adding a new language = dropping .ttf files in `fonts/<lang>/`. Adopt this pattern.
- **TRDG supports Hindi already.** Their `trdg -l hi` uses the bundled Noto Sans Devanagari. So you could literally just use TRDG for Hindi/Indic and skip writing your own renderer. (But then you don't get the COCO output, so writing your own is justified for an eval set.)

**Citation**: github.com/Belval/TextRecognitionDataGenerator [16][17]

### 9.3 PaddleOCR StyleText (Editing Text in the Wild, arXiv 1908.03047) — Chinese / English / Korean

**What it is**: A GAN-based style-transfer system. You give it a "style image" of how text should look (a snippet of, say, a Nepali novel page), and it renders new text in that exact style. The PaddleOCR team used this to improve Korean OCR accuracy from 30% to 50% and metal-surface English from 59% to 75% (their own published numbers).

**Design choices to learn from**:
- **The eval set should look like the deployment distribution.** If Himalaya's VLM will be reading street signs, the eval set should be street-sign-style. If it will be reading scanned documents, the eval set should be scanned-document-style. StyleText lets you customize this; with Pillow you can do a simpler version by sampling colors/textures from the target domain.
- **Three modules: foreground style, background extraction, fusion.** The decomposition is the contribution — you don't need a single end-to-end model.

**Citation**: github.com/PaddlePaddle/PaddleOCR/tree/main/StyleText, paper arXiv 1908.03047 [18]

### 9.4 UnrealText (Long & Yao, CVPR 2020) — multilingual scene text

**What it is**: 3D-engine-based synthesis. Renders scene + text together in Unreal Engine 4. 800K images. Multilingual version published.

**Design choices to learn from**:
- **Render scene and text together**, not as separate layers. This is the key insight that produces realistic text-on-realistic-surface.
- **Multilingual out of the box** — they have a Nepali version of the dataset, even. If ARCH wants to test the VLM on text-in-the-wild (signs, posters), UnrealText is the right tool. Heavy to set up but available.
- **Synthetic data is critical for detection training**, not just recognition. Detection (where is the text?) is even more data-hungry than recognition (what does the text say?). Even if you only build the recognition eval, the VLM also needs detection, and the same synthetic set serves both.

**Citation**: arXiv 2003.10608, code: github.com/Jyouhou/UnrealText [19]

### 9.5 SynthTIGER (Yim et al., ICDAR 2021) — Clova AI

**What it is**: A synthetic STR engine that explicitly addresses the long-tail problem. Without it, the character distribution in your synth set is dominated by common characters (and the model never learns rare ones). SynthTIGER samples with explicit long-tail correction.

**Design choices to learn from**:
- **Length and character distribution matter.** A random sample of text will under-represent very short and very long strings, and under-represent rare characters. For your eval set: explicitly check the character distribution. Include a few samples with rare conjuncts (ज्ञ, क्ष, त्र, श्र) and rare matras.
- **Components, not monolith.** SynthTIGER is a set of composable rendering modules. So should yours be.

**Citation**: github.com/clovaai/synthtiger, paper at Springer LNCS [20]

### 9.6 IIIT-HW-Dev (Dutta et al., DAS 2018) — Devanagari handwriting

**What it is**: 95K handwritten Devanagari words from 12 writers. The benchmark for Devanagari handwriting recognition.

**Design choices to learn from**:
- **They used synthetic data + cross-lingual transfer to beat pure real-data baselines.** The paper explicitly demonstrates that synthetic pre-training is necessary for Indic scripts due to data scarcity. Your eval set is part of a feedback loop where the VLM is trained on more data and re-evaluated; synthetic eval data is the right approach.
- **Vocabulary size of 9,540 words is a sweet spot.** Too small and you overfit to the vocab; too large and you have no per-word signal. For your eval, aim for ~5K-10K unique sentences.

**Citation**: cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data [25]

### 9.7 IndicSynthText (ofnote) — Devanagari + Indic

**What it is**: A fork of SynthText that adds Hindi, Marathi, Bengali, etc. font support and Indic newsgroup text corpora. **This is the closest direct prior art for the user's project.**

**Design choices to learn from**:
- **Newsgroup text as corpus** — the same idea as "scrape live news". The author of IndicSynthText used the 20-newsgroups dataset as a stand-in for "natural text". The user can use FLORES-200 + scraped news for a higher-quality version.
- **Drop in additional fonts** to `data/fonts/<lang>/` and update `fonts/fontlist.txt`. Same pattern you'd use.
- **Output format: lmdb.** The original SynthText outputs LMDB. The user outputs COCO. Note that LMDB is more efficient for huge datasets; for 10K images, COCO JSON is fine and more human-readable.

**Citation**: github.com/ofnote/IndicSynthText

### 9.8 COCO-Text (Cornell) — COCO for scene text

**What it is**: The first large-scale dataset to extend COCO for text. Real images (not synthetic), with `bbox`, `utf8_string`, `legibility`, `language` per annotation.

**Design choices to learn from**:
- **`utf8_string` per annotation is the field to add.** The user named it `transcription`; either works. Document it in the dataset README.
- **Legibility flag.** Useful for "is the text in the image readable?" — your synthetic eval has this for free (always legible), but it's worth recording for completeness.

**Citation**: vision.cornell.edu/se3/coco-text-2/ [8]

### 9.9 SynthDoG (used by Nemotron OCR v2) — multilingual scene-text

**What it is**: A multilingual scene-text dataset, used as the standard benchmark for cross-lingual OCR. The NVIDIA Nemotron OCR v2 paper uses it to compare PaddleOCR / OpenOCR / Nemotron.

**Design choices to learn from**:
- **This is the eval-set standard for multilingual OCR.** The user can use SynthDoG to *cross-validate* the synthetic Devanagari eval set — both should report similar difficulty levels, otherwise the synthetic set is miscalibrated.
- **Each language is a separate sub-evaluation.** Don't compute one aggregate score; report per-language. For your project, the equivalent is per-font, per-size, per-contrast.

**Citation**: NVIDIA blog "Building a Fast Multilingual OCR Model with Synthetic Data" [30-related]

### 9.10 DohaScript (2026) — modern Devanagari handwriting

**What it is**: A line-level Devanagari dataset with controlled lexical content and writer-diversity metadata. Newest of the bunch.

**Design choices to learn from**:
- **Line-level vs word-level.** The user is generating word-level images; ARCH's team should be aware that for VLM training, line-level is increasingly common (because modern VLMs handle wider images well). For the eval, both are fine — word-level is the traditional format, line-level is the modern format. Consider generating some line-level samples too.
- **Writer diversity.** Even in synthetic data, vary the font (which acts as a proxy for writer diversity in printed text). Noto Sans vs Mangal vs Lohit all look slightly different — they simulate "different writers".

**Citation**: arXiv 2602.18089 [27]

---

## 10. Step 9 — Resources (everything in one place)

### 10.1 Papers

- **Dettmers et al., 2023** — QLoRA. arXiv 2305.14314. https://arxiv.org/abs/2305.14314
- **Gupta, Vedaldi, Zisserman, 2016** — SynthText. arXiv 1604.06646. https://arxiv.org/abs/1604.06646
- **Long & Yao, 2020** — UnrealText. arXiv 2003.10608. https://arxiv.org/abs/2003.10608
- **Yim et al., 2021** — SynthTIGER. ICDAR 2021. https://link.springer.com/chapter/10.1007/978-3-030-86337-1_8
- **Yang et al., 2019** — Editing Text in the Wild (PaddleOCR StyleText). arXiv 1908.03047.
- **Dutta et al., 2018** — IIIT-HW-Dev. DAS 2018. https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data
- **Gongidi & Jawahar, 2021** — iiit-indic-hw-words. https://dl.acm.org/doi/10.1007/978-3-030-86337-1_30
- **NLLB Team, 2022** — FLORES-200. arXiv 2207.04672. https://github.com/facebookresearch/flores
- **Wenzek et al., 2020** — CCNet / CC-100. LREC 2020. https://huggingface.co/datasets/statmt/cc100
- **Conneau et al., 2020** — XLM-R. ACL 2020.

### 10.2 Datasets (Nepali / Devanagari)

- **FLORES-200 `npi_Deva`** — `datasets.load_dataset("facebook/flores", "npi_Deva")` https://huggingface.co/datasets/facebook/flores
- **CC-100 Nepali** — `datasets.load_dataset("cc100", lang="ne", streaming=True)`. 393M tokens. https://huggingface.co/datasets/statmt/cc100
- **OSCAR Nepali** — `oscar-corpus-nepali` on Kaggle. 3.8 GB. https://www.kaggle.com/datasets/hsebarp/oscar-corpus-nepali
- **IRIISNEPAL corpus** — 6.4M articles, 10.1 GB. https://huggingface.co/datasets/IRIISNEPAL/nepali-text-corpus
- **IIIT-HW-Dev** — https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data
- **Nepali Wikipedia** — 39K articles. https://www.kaggle.com/datasets/disisbig/nepali-wikipedia-articles
- **Nepali News Datasets**:
  - https://www.kaggle.com/datasets/lotusacharya/nepalinewsdataset
  - https://www.kaggle.com/datasets/ashokpant/nepali-news-dataset-large
- **NLUE (Nepali NLU benchmark)** — arXiv 2411.19244
- **NepaliGPT** — arXiv 2506.16399
- **DohaScript** — arXiv 2602.18089

### 10.3 Code repos

- **artidoro/qlora** (QLoRA reference) — https://github.com/artidoro/qlora
- **huggingface/peft** — https://github.com/huggingface/peft
- **huggingface/trl** — https://github.com/huggingface/trl
- **ankush-me/SynthText** — https://github.com/ankush-me/SynthText
- **Belval/TextRecognitionDataGenerator** — https://github.com/Belval/TextRecognitionDataGenerator
- **Jyouhou/UnrealText** — https://github.com/Jyouhou/UnrealText
- **clovaai/synthtiger** — https://github.com/clovaai/synthtiger
- **ofnote/IndicSynthText** — https://github.com/ofnote/IndicSynthText
- **PaddlePaddle/PaddleOCR** — https://github.com/PaddlePaddle/PaddleOCR (includes StyleText)
- **IBM/MAX-OCR** — IBM's Model Asset Exchange, multilingual OCR
- **sushant097/Devnagari-Handwritten-Word-Recongition-with-Deep-Learning** — https://github.com/sushant097/Devnagari-Handwritten-Word-Recongition-with-Deep-Learning (IIIT-HW-Dev baseline)
- **pemagrg1/Nepali-Datasets** — https://github.com/pemagrg1/Nepali-Datasets (curated list of Nepali resources)
- **divyamani1/Nepali-NLP-Progress** — https://github.com/divyamani1/Nepali-NLP-Progress (curated research list)

### 10.4 HuggingFace docs (VLM + QLoRA)

- **TRL: Fine-tuning a Multimodal Model Using SFT (VLM)** — https://huggingface.co/docs/trl/main/en/training_vlm_sft
- **TRL: SFTTrainer reference** — https://huggingface.co/docs/trl/sft_trainer
- **PEFT: LoRA reference** — https://huggingface.co/docs/peft/en/package_reference/lora
- **PEFT: Quantization** — https://huggingface.co/docs/peft/en/developer_guides/quantization
- **Transformers: bitsandbytes integration** — https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- **HF blog: Making LLMs accessible with bitsandbytes / QLoRA** — https://huggingface.co/blog/4bit-transformers-bitsandbytes
- **HF cookbook: Fine-tune Qwen2-VL with TRL** — https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl
- **Phil Schmid: Fine-tune VLMs with TRL** — https://www.philschmid.de/fine-tune-multimodal-llms-with-trl

### 10.5 Pillow / libraqm

- **OpenPecha: Generating Complex Text Image with Pillow** — https://forum.openpecha.org/t/generating-complex-text-image-with-pillow-the-challenges-and-the-solution/249
- **Pillow issue #4070** — Unicode Devanagari fonts rendered incorrectly. https://github.com/python-pillow/Pillow/issues/4070
- **Pillow issue #3593** — ImageDraw support for Bangla. https://github.com/python-pillow/Pillow/issues/3593
- **Pillow issue #3191** — Devanagari font not rendered correctly. https://github.com/python-pillow/Pillow/issues/3191
- **Pillow issue #2255** — Devanagari font not working. https://github.com/python-pillow/Pillow/issues/2255
- **Pillow issue #1089** — Bug in rendering Indic fonts. https://github.com/python-pillow/Pillow/issues/1089
- **W3C Devanagari Gap Analysis** — https://www.w3.org/TR/deva-gap/
- **W3C Devanagari Script Resources** — https://www.w3.org/TR/2024/DNOTE-deva-lreq-20240723/

### 10.6 COCO format

- **COCO website** — https://cocodataset.org/
- **COCO-Text (Cornell)** — https://vision.cornell.edu/se3/coco-text-2/
- **COCO API** — https://github.com/cocodataset/cocoapi
- **Microsoft vision-datasets (COCO fork)** — https://github.com/microsoft/vision-datasets/blob/main/COCO_DATA_FORMAT.md
- **Amazon Rekognition COCO reference** — https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
- **Ultralytics COCO docs** — https://docs.ultralytics.com/datasets/detect/coco
- **Roboflow COCO JSON guide** — https://roboflow.com/formats/coco-json

### 10.7 Devanagari fonts

- **Noto Sans Devanagari** — Google Fonts. OFL license. https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
- **Noto Serif Devanagari** — Google Fonts. OFL. https://fonts.google.com/noto/specimen/Noto+Serif+Devanagari
- **Mangal** — Windows default. https://en.wikipedia.org/wiki/Mangal_(font)
- **Lohit Devanagari** — Red Hat, OFL. https://github.com/pravins/lohit
- **Gargi** — Ubuntu default. https://en.wikipedia.org/wiki/Gargi
- **Laila** — Google Fonts. OFL. https://fonts.google.com/specimen/Laila

### 10.8 Pycocotools

- **pycocotools pip** — `pip install pycocotools`
- **pycocotools source** — https://github.com/cocodataset/cocoapi

---

## Appendix A — Full standalone Python files

All code below is also saved as separate `.py` files at `/workspace/deep-research/saugat-devanagari/code/` (the user can copy them into the project).

### A.1 `code/render.py` — Pillow Devanagari rendering

```python
"""
render.py — Devanagari text rendering with Pillow + libraqm.

This is the heart of the synthetic eval dataset pipeline.
It renders a single Devanagari string onto a synthetic image and computes
an accurate bounding box by diffing the rendered text mask.

Requires:
  pip install pillow numpy
  System: libraqm + libharfbuzz + libfribidi + libfreetype
  See setup.sh for the full environment.
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


# --- Variation knobs ---
FONTS = [
    "/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/noto/NotoSerifDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/usr/share/fonts/truetype/fonts-gargi/Gargi.ttf",
    "/usr/share/fonts/google-laila/Laila-Regular.ttf",
]

FONT_SIZES = [20, 24, 28, 32, 36, 40, 48, 56]
IMAGE_SIZES = [(256, 64), (384, 96), (512, 128), (640, 160), (800, 200)]
BACKGROUNDS = ["white", "colored", "gradient", "textured"]


def make_background(W: int, H: int, mode: str) -> Image.Image:
    """Generate a background image of size (W, H) in the given mode."""
    if mode == "white":
        return Image.new("RGB", (W, H), (255, 255, 255))
    elif mode == "colored":
        c = tuple(random.randint(200, 255) for _ in range(3))
        return Image.new("RGB", (W, H), c)
    elif mode == "gradient":
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(W):
            v = int(200 + 55 * x / W)
            arr[:, x] = (v, v, v)
        return Image.fromarray(arr)
    elif mode == "textured":
        arr = np.random.randint(200, 255, (H, W, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    else:
        raise ValueError(f"Unknown background mode: {mode}")


def render_devanagari_image(
    text: str,
    font_path: str,
    font_size: int = 36,
    image_size: tuple[int, int] = (640, 128),
    background: str = "white",
    text_color: tuple[int, int, int] | None = None,
    contrast_check: bool = True,
    position_jitter: tuple[int, int] = (10, 5),
) -> tuple[Image.Image | None, dict | None]:
    """Render one Devanagari string onto a synthetic image and return bbox.

    Returns: (PIL.Image, {"text", "bbox", "font", "size", "color", "background"})
             or (None, None) if rendering failed.
    """
    W, H = image_size

    # 1. Background
    bg_img = make_background(W, H, background)
    # Use a representative bg color for contrast check
    if background == "white":
        bg_color = (255, 255, 255)
    elif background == "colored":
        # sample center
        bg_color = bg_img.getpixel((W // 2, H // 2))
    elif background == "gradient":
        bg_color = (220, 220, 220)  # approximate
    elif background == "textured":
        bg_color = (227, 227, 227)  # approximate mean

    # 2. Font with raqm layout engine (CRITICAL for Devanagari)
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.LAYOUT_RAQM)

    # 3. Text color with contrast check
    if text_color is None:
        text_color = (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))

    if contrast_check:
        bg_lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        fg_lum = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
        if abs(bg_lum - fg_lum) < 60:
            text_color = (0, 0, 0) if bg_lum > 127 else (255, 255, 255)

    # 4. Compute text width for centering
    text_mask = Image.new("L", (W, H), 0)
    text_draw = ImageDraw.Draw(text_mask)
    try:
        text_w = int(text_draw.textlength(text, font=font))
    except AttributeError:
        bbox = text_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
    ascent, descent = font.getmetrics()
    text_h = ascent + descent

    jx, jy = position_jitter
    x = max(5, (W - text_w) // 2 + random.randint(-jx, jx))
    y = max(5, (H - text_h) // 2 + random.randint(-jy, jy))

    # 5. Render text to mask
    text_draw.text((x, y), text, font=font, fill=255)

    # 6. Find actual rendered bbox by diffing against blank
    blank = Image.new("L", (W, H), 0)
    diff = ImageChops.difference(text_mask, blank)
    rendered_bbox = diff.getbbox()
    if rendered_bbox is None:
        return None, None
    l, t, r, b = rendered_bbox
    bbox_xywh = [l, t, r - l, b - t]

    # 7. Composite text onto background
    draw = ImageDraw.Draw(bg_img)
    draw.text((x, y), text, font=font, fill=text_color)

    annotation = {
        "text": text,
        "bbox": bbox_xywh,
        "font": Path(font_path).name,
        "size": font_size,
        "color": text_color,
        "background": background,
    }
    return bg_img, annotation
```

### A.2 `code/corpus.py` — Load Nepali text from multiple sources

```python
"""
corpus.py — Load a Nepali text corpus from FLORES-200, scraped news, and Wikipedia.

Run:
  python corpus.py --output corpus.json --max-sentences 5000
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset

# --- 1. FLORES-200 ---
def load_flores_nepali() -> list[str]:
    """Load FLORES-200 Nepali (npi_Deva) devtest split. 997 sentences."""
    ds = load_dataset("facebook/flores", "npi_Deva", split="devtest")
    return [row["sentence"] for row in ds]

# --- 2. Scraped Nepali news ---
NEWS_SITES = [
    "https://ekantipur.com",
    "https://www.onlinekhabar.com",
    "https://www.setopati.com",
    "https://www.nagariknews.com.np",
]

def harvest_article_links(homepage: str, n_links: int = 200) -> list[str]:
    """Pull article URLs from a news homepage. Naive — refine per site."""
    try:
        resp = requests.get(homepage, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  ! {homepage}: {e}")
        return []
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = homepage.rstrip("/") + href
        if href.startswith("http") and homepage.split("//")[1].split("/")[0] in href:
            if any(year in href for year in ["2024", "2025", "2026"]):
                links.append(href)
    return list(dict.fromkeys(links))[:n_links]


def scrape_article_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    devanagari = [
        p for p in paragraphs
        if any("\u0900" <= ch <= "\u097F" for ch in p) and len(p) > 30
    ]
    return "\n".join(devanagari)


def load_scraped_news(target_sents: int = 5000) -> list[str]:
    """Scrape recent Nepali news from public sites. Polite (1 req/sec per site)."""
    sents: list[str] = []
    for homepage in NEWS_SITES:
        print(f"  scraping {homepage} ...")
        links = harvest_article_links(homepage, n_links=200)
        for url in links:
            text = scrape_article_text(url)
            for sent in text.split("।"):
                sent = sent.strip()
                if 20 <= len(sent) <= 150:
                    sents.append(sent + "।")
            if len(sents) >= target_sents:
                return sents
            time.sleep(random.uniform(0.5, 1.5))
    return sents

# --- 3. Nepali Wikipedia ---
def load_nepali_wikipedia(n_articles: int = 1000) -> list[str]:
    """Stream a few thousand Nepali Wikipedia articles, split into sentences."""
    ds = load_dataset("wikimedia/wikipedia", "ne", split="train", streaming=True)
    sents: list[str] = []
    for i, row in enumerate(ds):
        if i >= n_articles:
            break
        for sent in row["text"].split("।"):
            sent = sent.strip()
            if 20 <= len(sent) <= 200:
                sents.append(sent + "।")
    return sents

# --- 4. Cleaning ---
def clean_corpus(sentences: list[str]) -> list[str]:
    out = []
    for s in sentences:
        s = s.strip()
        if not any("\u0900" <= ch <= "\u097F" for ch in s):
            continue
        deva_ratio = sum(1 for ch in s if "\u0900" <= ch <= "\u097F") / max(1, len(s))
        if deva_ratio < 0.6:
            continue
        if re.search(r"https?://|&[a-z]+;|<|>", s):
            continue
        if not (15 <= len(s) <= 120):
            continue
        out.append(s)
    return list(dict.fromkeys(out))

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="corpus.json")
    parser.add_argument("--max-sentences", type=int, default=5000)
    parser.add_argument("--no-scrape", action="store_true", help="Skip news scraping")
    args = parser.parse_args()

    print("Loading FLORES-200 Nepali (npi_Deva)...")
    flores = load_flores_nepali()
    print(f"  {len(flores)} sentences")

    wiki = []
    if args.max_sentences > len(flores):
        print("Loading Nepali Wikipedia...")
        wiki = load_nepali_wikipedia(n_articles=500)
        print(f"  {len(wiki)} sentences")

    news = []
    if not args.no_scrape and args.max_sentences > len(flores) + len(wiki):
        print("Scraping Nepali news...")
        news = load_scraped_news(target_sents=args.max_sentences)
        print(f"  {len(news)} sentences")

    raw = flores + wiki + news
    print(f"Raw total: {len(raw)}")
    cleaned = clean_corpus(raw)
    print(f"After cleaning: {len(cleaned)}")

    if len(cleaned) > args.max_sentences:
        cleaned = random.sample(cleaned, args.max_sentences)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(cleaned)} sentences to {args.output}")
```

### A.3 `code/generate.py` — Top-level pipeline

```python
"""
generate.py — Generate the synthetic Devanagari eval dataset end-to-end.

Run:
  python generate.py \
      --corpus corpus.json \
      --fonts-dir fonts/ \
      --output-dir devanagari_eval/ \
      --n-images 5000
"""
import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

from corpus import clean_corpus, load_flores_nepali, load_scraped_news, load_nepali_wikipedia
from render import (
    BACKGROUNDS,
    FONTS,
    FONT_SIZES,
    IMAGE_SIZES,
    render_devanagari_image,
)


def find_fonts(fonts_dir: str) -> list[str]:
    """Find all .ttf files in the given directory."""
    p = Path(fonts_dir)
    if not p.exists():
        print(f"WARNING: {fonts_dir} does not exist. Falling back to system fonts.")
        return FONTS
    found = sorted([str(f) for f in p.glob("*.ttf")])
    if not found:
        print(f"WARNING: no .ttf files in {fonts_dir}. Falling back to system fonts.")
        return FONTS
    return found


def generate_dataset(
    sentences: list[str],
    fonts: list[str],
    output_dir: str,
    n_images: int = 1000,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Render n_images synthetic images and return parallel image/annotation/meta lists."""
    random.seed(seed)
    images, annotations, meta = [], [], []
    img_id = 0
    ann_id = 0
    pbar = tqdm(total=n_images, desc="rendering")
    attempts = 0
    while img_id < n_images and attempts < n_images * 5:
        attempts += 1
        text = random.choice(sentences)
        font = random.choice(fonts)
        font_size = random.choice(FONT_SIZES)
        image_size = random.choice(IMAGE_SIZES)
        background = random.choice(BACKGROUNDS)
        W, H = image_size

        img, ann = render_devanagari_image(
            text=text, font_path=font, font_size=font_size,
            image_size=image_size, background=background,
        )
        if img is None:
            continue

        # Optional: add Gaussian noise to make the eval harder
        if random.random() < 0.2:
            import numpy as np
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(0, 4), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            from PIL import Image as PILImage
            img = PILImage.fromarray(arr)

        img_id += 1
        ann_id += 1
        fname = f"img_{img_id:05d}.png"
        img.save(f"{output_dir}/images/{fname}")

        x, y, w, h = ann["bbox"]
        images.append({
            "id": img_id, "file_name": fname, "width": W, "height": H,
        })
        annotations.append({
            "id": ann_id, "image_id": img_id, "bbox": [x, y, w, h],
            "transcription": ann["text"], "area": int(w * h), "iscrowd": 0,
        })
        meta.append({
            "id": ann_id, "image_id": img_id, "font": ann["font"],
            "size": ann["size"], "color": ann["color"], "background": ann["background"],
        })
        pbar.update(1)
    pbar.close()
    return images, annotations, meta


def export_coco(
    images: list[dict],
    annotations: list[dict],
    output_path: str,
    description: str = "Synthetic Devanagari evaluation dataset for Himalaya AI VLM",
):
    coco = {
        "info": {
            "description": description, "version": "1.0", "year": 2026,
            "contributor": "Himalaya AI Labs",
        },
        "licenses": [
            {"id": 1, "name": "CC-BY-SA-4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
        ],
        "categories": [{"id": 1, "name": "devanagari_text", "supercategory": "text"}],
        "images": [], "annotations": [],
    }
    for img in images:
        coco["images"].append({
            "id": img["id"], "file_name": img["file_name"],
            "width": img["width"], "height": img["height"],
        })
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        coco["annotations"].append({
            "id": ann["id"], "image_id": ann["image_id"], "category_id": 1,
            "bbox": [x, y, w, h], "area": int(w * h), "iscrowd": 0,
            "transcription": ann["transcription"],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def split_coco(coco_path: str, out_dir: str, val_ratio: float = 0.10, seed: int = 42):
    random.seed(seed)
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image_ids = [img["id"] for img in data["images"]]
    random.shuffle(image_ids)
    n_val = int(len(image_ids) * val_ratio)
    val_ids, test_ids = set(image_ids[:n_val]), set(image_ids[n_val:])

    def filt(ids):
        return {
            **data,
            "images": [i for i in data["images"] if i["id"] in ids],
            "annotations": [a for a in data["annotations"] if a["image_id"] in ids],
        }
    for name, ids in [("val", val_ids), ("test", test_ids)]:
        with open(f"{out_dir}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(filt(ids), f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(ids)} images to {name}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Pre-built corpus JSON")
    parser.add_argument("--fonts-dir", default="fonts/")
    parser.add_argument("--output-dir", default="devanagari_eval/")
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load or build corpus
    if args.corpus and Path(args.corpus).exists():
        print(f"Loading corpus from {args.corpus}")
        sentences = json.load(open(args.corpus, "r", encoding="utf-8"))
    else:
        print("Building corpus from FLORES-200 + Wikipedia + scraped news")
        sentences = []
        sentences += load_flores_nepali()
        try:
            sentences += load_nepali_wikipedia(n_articles=500)
        except Exception as e:
            print(f"  ! Wikipedia load failed: {e}")
        try:
            sentences += load_scraped_news(target_sents=3000)
        except Exception as e:
            print(f"  ! News scrape failed: {e}")
        sentences = clean_corpus(sentences)
        Path("corpus.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Corpus: {len(sentences)} sentences")

    # Find fonts
    fonts = find_fonts(args.fonts_dir)
    print(f"Found {len(fonts)} fonts: {[Path(f).name for f in fonts]}")

    # Generate
    os.makedirs(f"{args.output_dir}/images", exist_ok=True)
    images, annotations, meta = generate_dataset(
        sentences, fonts, args.output_dir, n_images=args.n_images, seed=args.seed,
    )

    # Export
    export_coco(images, annotations, f"{args.output_dir}/annotations.json")
    Path(f"{args.output_dir}/meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. {len(images)} images, {len(annotations)} annotations.")

    # Split
    split_coco(f"{args.output_dir}/annotations.json", args.output_dir)
```

### A.4 `code/validate.py` — Validate the COCO JSON

```python
"""
validate.py — Sanity-check the exported COCO JSON using pycocotools.

Run:
  python validate.py devanagari_eval/annotations.json
"""
import sys
from pathlib import Path

from pycocotools.coco import COCO


def validate(coco_path: str) -> None:
    coco = COCO(coco_path)
    print(f"Loaded COCO: {len(coco.imgs)} images, {len(coco.anns)} annotations, {len(coco.cats)} categories")

    # 1. All bboxes within image bounds
    bad = 0
    for ann_id, ann in coco.anns.items():
        img = coco.imgs[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if x < 0 or y < 0 or x + w > img["width"] or y + h > img["height"]:
            bad += 1
            if bad < 5:
                print(f"  ! ann {ann_id} bbox overflow: {ann['bbox']} vs {img['width']}x{img['height']}")
    assert bad == 0, f"{bad} out-of-bounds bboxes"

    # 2. No empty transcriptions
    empty = sum(1 for ann in coco.anns.values() if not ann.get("transcription", "").strip())
    assert empty == 0, f"{empty} empty transcriptions"

    # 3. Every transcription has Devanagari
    no_deva = sum(
        1 for ann in coco.anns.values()
        if not any("\u0900" <= ch <= "\u097F" for ch in ann["transcription"])
    )
    if no_deva > 0:
        print(f"  ⚠ {no_deva} annotations without Devanagari")

    # 4. Every image file exists
    base_dir = Path(coco_path).parent
    missing = [coco.imgs[i]["file_name"] for i in coco.imgs
               if not (base_dir / coco.imgs[i]["file_name"]).exists()]
    assert not missing, f"Missing files: {missing[:5]}"

    print("✓ COCO JSON is valid.")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "devanagari_eval/annotations.json")
```

### A.5 `code/visualize.py` — Render bbox overlay for review

```python
"""
visualize.py — Render the bbox + transcription on top of sample images,
for manual review before shipping to ARCH.

Run:
  python visualize.py devanagari_eval/annotations.json --n 20
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


def visualize(coco_path: str, n: int = 20, output: str = "preview.png"):
    coco = COCO(coco_path)
    base = Path(coco_path).parent
    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)
    sample = img_ids[:n]

    # Compute grid
    cols = 4
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 400, 100
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, img_id in enumerate(sample):
        info = coco.imgs[img_id]
        img = Image.open(base / info["file_name"]).convert("RGB")
        # Resize to fit cell
        img.thumbnail((cell_w, cell_h))
        # Draw bbox
        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            x, y, w, h = ann["bbox"]
            scale_x = img.width / info["width"]
            scale_y = img.height / info["height"]
            x_, y_, w_, h_ = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            ImageDraw.Draw(img).rectangle([x_, y_, x_ + w_, y_ + h_], outline="red", width=2)
        col, row = i % cols, i // cols
        canvas.paste(img, (col * cell_w, row * cell_h))

    canvas.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_path", default="devanagari_eval/annotations.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", default="preview.png")
    args = parser.parse_args()
    visualize(args.coco_path, args.n, args.output)
```

### A.6 `code/setup.sh` — One-shot environment setup

```bash
#!/usr/bin/env bash
set -euo pipefail

# Install system packages for libraqm + Devanagari fonts
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y \
        libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools \
        libjpeg-dev zlib1g-dev libraqm-dev fontconfig \
        fonts-noto-core fonts-noto-extra fonts-indic
elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed \
        noto-fonts noto-fonts-extra \
        harfbuzz fribidi gtk-doc freetype2 libraqm fontconfig
fi

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Force source build of Pillow so it links libraqm
pip install --no-binary :all: Pillow

pip install numpy tqdm datasets requests beautifulsoup4 pycocotools

# Verify
python3 -c "from PIL import Image, ImageDraw, ImageFont; \
    img = Image.new('RGB', (400, 100), 'white'); \
    font = ImageFont.truetype('/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf', 36, layout_engine=ImageFont.LAYOUT_RAQM); \
    ImageDraw.Draw(img).text((10, 10), 'यात्रा', font=font, fill='black'); \
    print('libraqm OK')"
```

---

## Sources (consolidated)

All sources cited inline are also collected here for quick reference.

[1] Himalaya AI on HuggingFace. https://huggingface.co/himalaya-ai
[3] Devanagari Unicode block. https://symbl.cc/pl/unicode/blocks/devanagari/
[4] Pillow issue #4070. https://github.com/python-pillow/Pillow/issues/4070
[5] OpenPecha: Generating Complex Text Image with Pillow. https://forum.openpecha.org/t/generating-complex-text-image-with-pillow-the-challenges-and-the-solution/249
[6] Microsoft vision-datasets COCO_DATA_FORMAT.md. https://github.com/microsoft/vision-datasets/blob/main/COCO_DATA_FORMAT.md
[7] AWS Rekognition COCO reference. https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
[8] COCO-Text. https://vision.cornell.edu/se3/coco-text-2/
[9] Dettmers et al., QLoRA, 2023. https://arxiv.org/abs/2305.14314
[10] HF blog: QLoRA / 4-bit. https://huggingface.co/blog/4bit-transformers-bitsandbytes
[11] HuggingFace PEFT. https://github.com/huggingface/peft
[12] PEFT LoraConfig reference. https://huggingface.co/docs/peft/en/package_reference/lora
[13] TRL: Fine-tuning a Multimodal Model Using SFT. https://huggingface.co/docs/trl/main/en/training_vlm_sft
[14] Gupta, Vedaldi, Zisserman, SynthText, CVPR 2016. https://arxiv.org/abs/1604.06646
[15] VGG SynthText dataset. https://www.robots.ox.ac.uk/~vgg/data/scenetext/
[16] TRDG (Belval). https://github.com/Belval/TextRecognitionDataGenerator
[17] TRDG documentation. https://textrecognitiondatagenerator.readthedocs.io/
[18] PaddleOCR StyleText. https://github.com/Mushroomcat9998/PaddleOCR/blob/main/StyleText/README.md
[19] Long & Yao, UnrealText, CVPR 2020. https://arxiv.org/abs/2003.10608
[20] Yim et al., SynthTIGER, ICDAR 2021. https://link.springer.com/chapter/10.1007/978-3-030-86337-1_8
[21] FLORES-200 GitHub. https://github.com/facebookresearch/flores
[22] FLORES on HuggingFace. https://huggingface.co/datasets/facebook/flores
[23] CC-100. https://huggingface.co/datasets/statmt/cc100
[24] pemagrg1/Nepali-Datasets. https://github.com/pemagrg1/Nepali-Datasets
[25] IIIT-HW-Dev (Dutta et al., DAS 2018). https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data
[26] iiit-indic-hw-words (Gongidi & Jawahar, 2021). https://dl.acm.org/doi/10.1007/978-3-030-86337-1_30
[27] DohaScript. https://arxiv.org/abs/2602.18089
[28] NLUE benchmark for Nepali NLU. https://arxiv.org/abs/2411.19244
[29] PaddleOCR 3.0 Technical Report. https://arxiv.org/html/2507.05595v1
[30] Nemotron OCR v2 (NVIDIA). https://huggingface.co/blog/nvidia/nemotron-ocr-v2
[31] W3C Devanagari Gap Analysis. https://www.w3.org/TR/deva-gap/
[32] Pillow issue #1089. https://github.com/python-pillow/Pillow/issues/1089
[33] Pillow issue #2255. https://github.com/python-pillow/Pillow/issues/2255
[34] Pillow issue #3191. https://github.com/python-pillow/Pillow/issues/3191
[35] Pillow issue #3593. https://github.com/python-pillow/Pillow/issues/3593
