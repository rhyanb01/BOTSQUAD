# script2_gaussian_blur.py

import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_gaussian_to_processed(original_filename, radius=2.0, folder="processing_image"):
    input_path = os.path.join(folder, os.path.basename(original_filename))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Cannot find {input_path}. Run the grayscale step first.")

    img = Image.open(input_path).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

    name, _ = os.path.splitext(os.path.basename(original_filename))
    output_filename = f"{name}+gauss.jpeg"
    output_path = os.path.join(folder, output_filename)

    os.makedirs(folder, exist_ok=True)
    blurred.save(output_path, format="JPEG", quality=95)

    print(f"✅ Gaussian-blurred image saved to: {output_path}")
    return output_path

# --- MAIN ---
if __name__ == "__main__":
    # Change only the name to match the output from step 1
    grayscale_image = "CNN greyscale.jpg"
    blur_radius = 2.0

    try:
        output = apply_gaussian_to_processed(grayscale_image, radius=blur_radius)

        img = Image.open(output)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Gaussian Blurred Image")
        plt.show()
    except FileNotFoundError as e:
        print(e)
