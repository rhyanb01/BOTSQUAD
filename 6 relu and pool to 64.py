# relu_and_pool_next.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def relu_and_pool_2x2(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Cannot find: {input_path}")

    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    h, w = arr.shape
    if h < 2 or w < 2:
        raise ValueError(f"Image too small for 2x2 pooling: {h}x{w}")

    # 1) ReLU
    relu = np.maximum(0, arr)

    # 2) MaxPooling 2x2 → halves each dimension
    pooled_h, pooled_w = h // 2, w // 2
    # trim to even size if needed
    relu = relu[:pooled_h*2, :pooled_w*2]

    # reshape trick for fast 2x2 max
    relu_reshaped = relu.reshape(pooled_h, 2, pooled_w, 2)
    pooled = relu_reshaped.max(axis=(1,3))

    pooled = np.clip(pooled, 0, 255).astype(np.uint8)

    folder = os.path.dirname(input_path)
    name, _ = os.path.splitext(os.path.basename(input_path))
    out_path = os.path.join(folder, f"{name}+relu+pool64.jpeg")

    Image.fromarray(pooled).save(out_path, format="JPEG", quality=95)
    print(f"✅ ReLU + 2x2 MaxPool saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    input_file = 'processing_image\CNN+gauss+sobel+pool128+sharpen.jpeg'
    result_path = relu_and_pool_2x2(input_file)

    # Optional preview
    plt.imshow(Image.open(result_path), cmap="gray")
    plt.axis("off")
    plt.show()
