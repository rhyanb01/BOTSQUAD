import os
import json
import joblib
import pygame
import numpy as np
from PIL import Image, ImageOps

# -------- Config (must match training) --------
MODEL_PATH = "number_model.pkl"
META_PATH = "number_model_meta.json"

WIDTH, HEIGHT = 400, 400   # drawing canvas
UI_HEIGHT = 100
BG_COLOR = (255, 255, 255)
INK_COLOR = (0, 0, 0)
ERASE_COLOR = BG_COLOR
START_BRUSH = 16
MIN_BRUSH, MAX_BRUSH = 2, 64

def load_model_and_meta():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Trained model not found. Run train_number_classifier.py first.\n"
            f"Missing {MODEL_PATH} or {META_PATH}."
        )
    clf = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return clf, meta

def surface_to_numpy(surface, width, height):
    """Extract the drawing region as a (H, W, 3) numpy array."""
    sub = pygame.Surface((width, height))
    sub.blit(surface, (0, 0), area=pygame.Rect(0, 0, width, height))
    # Convert to string buffer, then to array
    arr = pygame.surfarray.array3d(sub)  # (W,H,3) in pygame
    arr = np.transpose(arr, (1, 0, 2))   # to (H,W,3)
    return arr

def preprocess_for_model(arr_rgb, img_size, white_threshold):
    """
    Same steps as training:
      - to grayscale
      - invert
      - autocrop ink
      - pad to square
      - resize to img_size
      - normalize, flatten
    """
    # to PIL image
    img = Image.fromarray(arr_rgb.astype(np.uint8))
    img_gray = ImageOps.grayscale(img)
    img_inv = ImageOps.invert(img_gray)

    arr = np.array(img_inv)
    mask = arr > (255 - white_threshold)
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        cropped = img_inv
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        cropped = img_inv.crop((x_min, y_min, x_max, y_max))

    w, h = cropped.size
    side = max(w, h)
    padded = Image.new("L", (side, side), color=0)
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    padded.paste(cropped, (paste_x, paste_y))

    small = padded.resize((img_size, img_size), Image.BILINEAR)
    vec = np.asarray(small, dtype=np.float32) / 255.0
    return vec.flatten()

def draw_line_with_brush(surf, color, start, end, radius):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    dist = max(1, int((dx*dx + dy*dy) ** 0.5))
    for i in range(dist + 1):
        t = i / dist
        x = int(x1 + t * dx)
        y = int(y1 + t * dy)
        pygame.draw.circle(surf, color, (x, y), radius)

def draw_ui(screen, font, big_font, prediction_text, prob_text, brush_size):
    info_bg = (245, 245, 245)
    pygame.draw.rect(screen, info_bg, (0, HEIGHT, WIDTH, UI_HEIGHT))

    lines = [
        "Left-draw | Right-erase | C=clear | +/- change brush | Space=predict | Esc=quit",
        f"Brush: {brush_size}px",
        prediction_text,
        prob_text
    ]
    y = HEIGHT + 8
    for line in lines:
        surf = font.render(line, True, (20, 20, 20))
        screen.blit(surf, (10, y))
        y += surf.get_height() + 6

def main():
    clf, meta = load_model_and_meta()
    img_size = int(meta["img_size"])
    white_threshold = int(meta["white_threshold"])
    classes = meta["classes"]

    pygame.init()
    pygame.display.set_caption("Number Predictor")
    screen = pygame.display.set_mode((WIDTH, HEIGHT + UI_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    big_font = pygame.font.SysFont(None, 28)

    # Canvas
    screen.fill(BG_COLOR)

    brush_size = START_BRUSH
    drawing = False
    erasing = False
    last_pos = None

    prediction_text = "Prediction: (draw and press Space)"
    prob_text = ""

    running = True
    while running:
        clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y < HEIGHT:
                    if event.button == 1:
                        drawing = True
                        erasing = False
                        last_pos = (x, y)
                        pygame.draw.circle(screen, INK_COLOR, (x, y), brush_size)
                    elif event.button == 3:
                        drawing = False
                        erasing = True
                        last_pos = (x, y)
                        pygame.draw.circle(screen, ERASE_COLOR, (x, y), brush_size)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                elif event.button == 3:
                    erasing = False
                last_pos = None

            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                if y < HEIGHT and last_pos is not None:
                    if drawing:
                        draw_line_with_brush(screen, INK_COLOR, last_pos, (x, y), brush_size)
                        last_pos = (x, y)
                    elif erasing:
                        draw_line_with_brush(screen, ERASE_COLOR, last_pos, (x, y), brush_size)
                        last_pos = (x, y)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    pygame.draw.rect(screen, BG_COLOR, (0, 0, WIDTH, HEIGHT))
                    prediction_text = "Prediction: (draw and press Space)"
                    prob_text = ""
                elif event.key == pygame.K_SPACE:
                    # Grab current drawing and predict
                    arr = surface_to_numpy(screen, WIDTH, HEIGHT)
                    vec = preprocess_for_model(arr, img_size, white_threshold).reshape(1, -1)
                    # Some sklearn models may not support predict_proba; RF does.
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(vec)[0]
                        idx = int(np.argmax(proba))
                        pred = clf.classes_[idx]
                        conf = float(proba[idx])
                    else:
                        pred = clf.predict(vec)[0]
                        conf = None
                    prediction_text = f"Prediction: {pred}"
                    prob_text = f"Confidence: {conf:.2f}" if conf is not None else "(no proba)"
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    brush_size = min(MAX_BRUSH, brush_size + 1)
                elif event.key == pygame.K_MINUS:
                    brush_size = max(MIN_BRUSH, brush_size - 1)

        # UI
        draw_ui(screen, font, big_font, prediction_text, prob_text, brush_size)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
