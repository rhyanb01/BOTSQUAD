import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pooling_layer_to_128x128(input_path):
    """
    Simulates a MaxPooling layer by downsampling the given image to 128x128
    using the maximum pixel value in each block.
    Saves output as <original_name>+pool128.jpeg in the same folder.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Cannot find file: {input_path}")

    # Extract folder + filename info
    folder = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)

    # Load grayscale image
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # Determine pooling region sizes
    pool_h = h // 128
    pool_w = w // 128

    if pool_h == 0 or pool_w == 0:
        raise ValueError(f"Image is too small ({h}x{w}) to pool down to 128x128.")

    # Initialize pooled output
    pooled = np.zeros((128, 128), dtype=np.float32)

    # Perform block-wise max pooling
    for i in range(128):
        for j in range(128):
            block = arr[i*pool_h:(i+1)*pool_h, j*pool_w:(j+1)*pool_w]
            pooled[i, j] = np.max(block)

    # Normalize & save
    pooled = np.clip(pooled, 0, 255).astype(np.uint8)
    out_path = os.path.join(folder, f"{name}+pool128.jpeg")
    Image.fromarray(pooled).save(out_path, format="JPEG", quality=95)

    print(f"✅ Pooled image saved to: {out_path}")
    return out_path


# --- MAIN ---
if __name__ == "__main__":
    # Your specified file
    input_image_path = r"processing_image\CNN+gauss+sobel.jpeg"

    result_path = pooling_layer_to_128x128(input_image_path)

    # Show the pooled image
    img = Image.open(result_path)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
