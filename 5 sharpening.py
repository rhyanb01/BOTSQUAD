import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_sharpening_kernel(input_path):
    """
    Applies a sharpening convolution kernel to the given pooled image
    and saves the result with '+sharpen.jpeg' added to the filename.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Cannot find: {input_path}")

    # Load grayscale image
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Define sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Pad the image to handle borders
    padded = np.pad(arr, pad_width=1, mode='edge')

    # Perform 2D convolution manually
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            region = padded[i:i+3, j:j+3]
            out[i, j] = np.sum(region * kernel)

    # Clip pixel values to valid range
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Build output path
    folder = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(folder, f"{name}+sharpen.jpeg")

    # Save and show
    Image.fromarray(out).save(out_path, format="JPEG", quality=95)
    print(f"✅ Sharpened image saved to: {out_path}")
    return out_path


# --- MAIN ---
if __name__ == "__main__":
    input_file = r"processing_image\CNN+gauss+sobel+pool128.jpeg"
    result_path = apply_sharpening_kernel(input_file)

    # Visualize
    img = Image.open(result_path)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
