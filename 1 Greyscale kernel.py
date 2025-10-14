# script1_grayscale.py

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from tkinter import Tk, filedialog

def convolve(img, kernel):
    width, height = img.size
    pixels = img.load()

    for py in tqdm(range(height), desc="Processing image", ncols=80):
        for px in range(width):
            r, g, b = img.getpixel((px, py))
            v = np.array([[r], [g], [b]])
            res = np.dot(kernel, v)
            tr, tg, tb = map(lambda x: min(int(x), 255), res.flatten())
            pixels[px, py] = (tr, tg, tb)

    return img

def save_processed_image(img, original_filename):
    folder_name = "processing_image"
    os.makedirs(folder_name, exist_ok=True)
    base_name = os.path.basename(original_filename)
    name, ext = os.path.splitext(base_name)
    save_path = os.path.join(folder_name, f"CNN greyscale.jpg")
    img.save(save_path)
    print(f"✅ Image saved to: {save_path}")
    return save_path

# --- MAIN ---
if __name__ == "__main__":
    # Open file dialog to select an image
    Tk().withdraw()
    input_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if not input_path:
        print("❌ No file selected.")
        exit()

    img = Image.open(input_path).convert("RGB")
    grayscale_kernel = np.ones((3, 3)) * (1/3)

    processed_img = convolve(img, grayscale_kernel)

    plt.imshow(processed_img)
    plt.axis('off')
    plt.title("Grayscale Image")
    plt.show()

    save_processed_image(processed_img, input_path)
