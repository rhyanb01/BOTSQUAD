import os
from PIL import Image

# === CONFIG ===
folder = r"processing_image"
input_name = "CNN+gauss+sobel+pool128+sharpen+relu+pool64"  # no extension
target_size = (32, 32)


output_name = f"CNN pool32.png"
output_path = os.path.join(folder, output_name)

# Locate the source image by trying common extensions
candidates = [os.path.join(folder, f"{input_name}{ext}") for ext in (".jpeg", ".jpg", ".png")]
src_path = next((p for p in candidates if os.path.exists(p)), None)
if src_path is None:
    raise FileNotFoundError(
        f"Could not find source image. Tried:\n- " + "\n- ".join(candidates)
    )

# Open, convert to grayscale (or keep RGB if you prefer), downscale, save
img = Image.open(src_path).convert("L")  # change to "RGB" if you want color
small = img.resize(target_size, Image.LANCZOS)
small.save(output_path, format="PNG", optimize=True)

print(f"✅ Saved 32×32 image to: {output_path}")
