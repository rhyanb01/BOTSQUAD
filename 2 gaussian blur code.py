import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_gaussian_to_processed(original_filename, radius=100.0, folder="processing_image"):
    """
    Applies Gaussian blur to a grayscale image already stored in 'processing_image/'.
    Saves the result as '<image_name>+gauss.jpeg' in the same folder.
    """
    # Build the path to the grayscale image in the folder
    input_path = os.path.join(folder, os.path.basename(original_filename))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Cannot find {input_path}. Make sure the grayscale image exists first.")

    # Load and ensure RGB for saving
    img = Image.open(input_path).convert("RGB")

    # Apply Gaussian blur
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Build output filename and path
    name, _ = os.path.splitext(os.path.basename(original_filename))
    output_filename = f"{name}+gauss.jpeg"
    output_path = os.path.join(folder, output_filename)

    # Save blurred image
    os.makedirs(folder, exist_ok=True)
    blurred.save(output_path, format="JPEG", quality=95)

    print(f"✅ Gaussian-blurred image saved to: {output_path}")
    return output_path


# --- MAIN ---
if __name__ == "__main__":
    input_image = "CNN.jpg"  # original file name (used to locate grayscale version)
    blur_radius = 2.0        # adjust to control strength of blur

    output = apply_gaussian_to_processed(input_image, radius=blur_radius)

    # (Optional) Display the blurred image
    img = Image.open(output)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

