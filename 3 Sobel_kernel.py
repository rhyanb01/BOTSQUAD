import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def sobel_on_gaussian(original_filename, folder="processing_image", threshold=None):
    """
    Applies Sobel edge detection to the *Gaussian-blurred* image:
        processing_image/<name>+gauss.jpeg
    and saves the result as:
        processing_image/<name>+gauss+sobel.jpeg

    Args:
        original_filename (str): base original name, e.g. "CNN.jpg"
        folder (str): directory containing the +gauss image
        threshold (int|None): optional 0..255 for binary edges; None keeps gradient magnitude
    """
    name, _ = os.path.splitext(os.path.basename(original_filename))
    gauss_path = os.path.join(folder, f"{name}+gauss.jpeg")
    if not os.path.exists(gauss_path):
        raise FileNotFoundError(f"❌ {gauss_path} not found. Create the Gaussian image first.")

    # Load blurred (convert to grayscale for Sobel)
    img = Image.open(gauss_path).convert("L")
    I = np.array(img, dtype=np.float32)

    # Sobel kernels
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Pad with reflect to handle borders
    P = np.pad(I, 1, mode="reflect")

    # Fast 3x3 convolution (vectorized)
    def conv3(P, K):
        return (
            K[0,0]*P[:-2, :-2] + K[0,1]*P[:-2, 1:-1] + K[0,2]*P[:-2, 2:] +
            K[1,0]*P[1:-1, :-2] + K[1,1]*P[1:-1, 1:-1] + K[1,2]*P[1:-1, 2:] +
            K[2,0]*P[2:, :-2] + K[2,1]*P[2:, 1:-1] + K[2,2]*P[2:, 2:]
        )

    Gx = conv3(P, Kx)
    Gy = conv3(P, Ky)
    mag = np.hypot(Gx, Gy)

    # Normalize to 0..255
    if mag.max() > 0:
        mag = mag / mag.max() * 255.0
    mag = np.clip(mag, 0, 255)

    # Optional threshold -> binary edges
    out_arr = (mag if threshold is None else (mag >= threshold).astype(np.uint8) * 255).astype(np.uint8)

    out_path = os.path.join(folder, f"{name}+gauss+sobel.jpeg")
    Image.fromarray(out_arr).save(out_path, format="JPEG", quality=95)
    print(f"✅ Sobel edges saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    original = "CNN.jpg"           # base name
    # threshold=None -> grayscale magnitude; set e.g. 100 for binary edges
    result_path = sobel_on_gaussian(original, threshold=None)

    # (Optional) preview
    plt.imshow(Image.open(result_path), cmap="gray")
    plt.axis("off")
    plt.show()
