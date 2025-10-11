import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

def convolve(img, kernel):
    width, height = img.size
    pixels = img.load()

    for py in tqdm(range(height), desc="Processing image", ncols=80):
        for px in range(width):
            r, g, b = img.getpixel((px, py))

            v = np.array([[r], [g], [b]])
            res = np.dot(kernel, v)

            tr, tg, tb = int(res[0, 0]), int(res[1, 0]), int(res[2, 0])

            # Keep your original clamp logic
            if tr > 255:
                tr = 255
            if tg > 255:
                tg = 255
            if tb > 255:
                tb = 255

            pixels[px, py] = (tr, tg, tb)

    return img

def save_processed_image(img, original_filename):
    """
    Saves the processed image into a folder named 'processing_image'.
    Creates the folder if it doesn't exist.
    """
    folder_name = "processing_image"
    os.makedirs(folder_name, exist_ok=True)  # make folder if not there

    # extract the base filename (without path)
    base_name = os.path.basename(original_filename)
    save_path = os.path.join(folder_name, base_name)

    img.save(save_path)
    print(f"✅ Image saved to: {save_path}")

# --- MAIN ---
input_path = 'CNN.jpg'
img = Image.open(input_path)
grayscale = np.ones((3, 3)) * (1/3)

processed_img = convolve(img, grayscale)

# Show and save
plt.imshow(processed_img)
plt.axis('off')
plt.show()

save_processed_image(processed_img, input_path)
