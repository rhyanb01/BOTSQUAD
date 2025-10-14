import os
import numpy as np
from PIL import Image
import csv

# === CONFIG ===
image_path = r"processing_image\CNN pool32.png"
output_csv = r"processing_image\image_pixels.csv"
mode = "L"   # "L" = grayscale, "RGB" = color

# === ASK FOR LABEL ===
label = input("Enter label for this image (e.g. apple, car, etc.): ").strip()

# === LOAD AND CONVERT IMAGE ===
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

img = Image.open(image_path).convert(mode)
arr = np.array(img)

# === FLATTEN PIXELS ===
if mode == "L":
    flat = arr.flatten()               # grayscale: 1 value per pixel
else:
    flat = arr.reshape(-1, 3).flatten()  # RGB: 3 values per pixel

# === NORMALIZE TO 0..1 (optional) ===
flat = flat / 255.0

# === COMBINE LABEL + PIXELS ===
row = [label] + flat.tolist()

# === SAVE / APPEND TO CSV ===
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

file_exists = os.path.isfile(output_csv)

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    # Optionally, write header only once
    if not file_exists:
        print("🆕 Creating new CSV file.")
    writer.writerow(row)

print(f"✅ Saved labeled data ({len(flat)} pixels) to:\n{output_csv}")
print(f"🟢 Label: {label}")
