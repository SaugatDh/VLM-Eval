import os
import re
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont,ImageFilter
from datasets import load_dataset

# -------------------------------------------------------------------------
# 1. Dataset Loading & Advanced Filtering (Hugging Face)
# -------------------------------------------------------------------------
def clean_and_validate_devanagari(text):
    if not text:
        return None
        
    # Step A: Strip out non-printable, hidden, or unsupported characters entirely
    # This keeps Devanagari, spaces, standard digits, and common punctuation
    cleaned_text = re.sub(r'[^\u0900-\u097F\s0-9.,\-?।!()]', '', text).strip()
    
    if not cleaned_text:
        return None

    # Step B: Check if the remaining text is mostly Devanagari characters
    devanagari_chars = re.findall(r'[\u0900-\u097F]', cleaned_text)
    
    # Count only non-whitespace characters for an accurate ratio check
    non_whitespace_len = len(re.sub(r'\s', '', cleaned_text))
    
    if non_whitespace_len > 0 and (len(devanagari_chars) / non_whitespace_len) > 0.7:
        return cleaned_text
    
    return None


print("Streaming sentences from Hugging Face...")
ds = load_dataset("himalaya-ai/cc100-nepali", split="train", streaming=True)
sentences = []

for row in ds:
    for line in row["text"].split("\n"):
        line = line.strip()
        
        # Enforce character length constraints
        if 10 <= len(line) <= 100:
            cleaned_line = clean_and_validate_devanagari(line)
            if cleaned_line:
                sentences.append(cleaned_line)
                
        if len(sentences) >= 500:
            break
    if len(sentences) >= 500:
        break

print(f"Loaded {len(sentences)} pristine, tofu-free Nepali sentences.") 

# -------------------------------------------------------------------------
# 2. Image Generation Helper Functions
# -------------------------------------------------------------------------

def get_background(width, height, mode="white"):
    """Generates a base background image depending on the mode selected."""
    if mode == "white":
        color = (255, 255, 255)
    elif mode == "colored":
        color = (255, 200, 150)  # light orange
    elif mode == "gray":
        color = (150, 150, 150)  # darker gray
    else:
        color = (255, 255, 255)
        
    return Image.new("RGB", (width, height), color=color)

def add_blur(img, radius=0.0):
    """Apply Gaussian blur for out-of-focus effect."""
    if radius > 0:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img

def add_noise(img, intensity=0.05):
    """Applies normal distribution noise over the PIL Image. Add freckles like noise"""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_rotation(img, max_degrees=3.0):
    """Slight rotation for skewed text."""
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

def add_low_contrast(img, probability=0.3):
    """Make text lighter/fainter."""
    if random.random() < probability:
        # Use lighter text color instead
        return img
    return img

def add_degradation(img):
    """Simulate worn/degraded print."""
    arr = np.array(img)
    from scipy import ndimage
    arr = ndimage.binary_erosion(arr < 200).astype(np.uint8) * 255
    return Image.fromarray(arr)

def render_image(text, font_path, font_size, default_img_size=(400, 100)):
    """Renders text randomly onto a generated background with noise.
    Dynamically adjusts canvas size if the text length exceeds default bounds.
    """
    bg_mode = random.choice(["white", "colored", "gray"])
    
    # Safely load the font, fallback to default if font path is broken
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        
    # Create a temporary canvas to accurately measure true text bounding box metrics first
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top

    # Dynamic canvas sizing: Ensure long text strings don't clip off borders
    img_w = max(default_img_size[0], w + 40)  # Includes a 40px buffer padding
    img_h = max(default_img_size[1], h + 40)

    img = get_background(img_w, img_h, mode=bg_mode)
    draw = ImageDraw.Draw(img)

    max_x = max(0, img_w - w)
    max_y = max(0, img_h - h)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Offset drawing relative to font metrics anchoring
    draw.text((x - left, y - top), text, font=font, fill=(0, 0, 0))
    img = add_noise(img, intensity=0.05)
    blur_intensity = random.choice([0, 0, 0, 1.0])
    img = add_blur(img, radius=blur_intensity)

    # Random hard cases
    if random.random() < 0.4:  # 40% rotated
        img = add_rotation(img, max_degrees=2.0)

    # 20% lighter text (low contrast)
    if random.random() < 0.2:
        draw.text((x - left, y - top), text, font=font, fill=(150, 150, 150))
    else:
        draw.text((x - left, y - top), text, font=font, fill=(0, 0, 0))

    return img, [x, y, w, h], bg_mode, (img_w, img_h)


# -------------------------------------------------------------------------
# 3. Initialization & Config
# -------------------------------------------------------------------------

output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

FONTS = [
    "/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/noto/NotoSerifDevanagari-Regular.ttf",
    "/usr/share/fonts/ibm-plex-sans-devanagari/IBMPlexSansDevanagari-Regular.ttf",
    "/usr/share/fonts/anek-devanagari/AnekDevanagari[wdth,wght].ttf",
    "/usr/share/fonts/lohit-devanagari/Lohit-Devanagari.ttf",
    "/usr/share/fonts/tiro-devanagari-hindi/TiroDevanagariHindi-Regular.ttf",
]

# COCO dataset skeleton structure
coco_dataset = {
    "info": {
        "year": 2026,
        "version": "1.0",
        "description": "Synthetic Clean Devanagari Text Dataset",
        "date_created": "2026-06-06"
    },
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "text", "supercategory": "character"}
    ]
}

# -------------------------------------------------------------------------
# 4. Main Generation Loop
# -------------------------------------------------------------------------

epochs = 100  # Generates 100 images from your pool of 500 clean sentences
for idx in range(1, epochs + 1):
    file_name = f"devanagari_{idx}.png"
    img_path = os.path.join(output_dir, file_name)
    
    text = random.choice(sentences)
    font_file = random.choice(FONTS)
    font_size = int(random.choice(np.arange(14, 38, 4)))  # Safe size bounds for sentences
    
    # Render using the adaptive canvas method
    img, bbox, bg_mode, final_size = render_image(
        text=text,
        font_path=font_file,
        font_size=font_size,
        default_img_size=(500, 120) 
    )
    img.save(img_path)

    # Append image metadata to COCO instances using dynamic target sizing
    coco_dataset["images"].append({
        "id": idx,
        "width": final_size[0],
        "height": final_size[1],
        "file_name": file_name
    })

    # Convert bbox back to standard COCO polygon coordinates format
    bx, by, bw, bh = bbox
    polygon = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
    
    # Append structured annotation metadata entry
    coco_dataset["annotations"].append({
        "id": 1000 + idx,
        "image_id": idx,
        "category_id": 1,
        "bbox": bbox,
        "area": float(bw * bh),
        "segmentation": [polygon],
        "iscrowd": 0,
        "transcription": text,
        "metadata": {
            "font_size": font_size,
            "bg_mode": bg_mode,
            "font_used": os.path.basename(font_file)
        }
    })

# Save the final structured COCO file
with open(os.path.join(output_dir, "annotations.json"), "w", encoding="utf-8") as f:
    json.dump(coco_dataset, f, indent=2, ensure_ascii=False)

print(f"Successfully generated {epochs} images and annotations.json in '{output_dir}/' folder.")