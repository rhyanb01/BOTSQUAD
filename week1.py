import os
import re
import csv
import sys
import pygame

# ---------- Config ----------
WIDTH, HEIGHT = 400, 400          # Canvas size
UI_HEIGHT = 80                    # Space at bottom for instructions/label
BG_COLOR = (255, 255, 255)        # White background
INK_COLOR = (0, 0, 0)             # Black drawing color
ERASE_COLOR = BG_COLOR            # Eraser uses background color
START_BRUSH = 16                  # Initial brush radius (px)
MIN_BRUSH, MAX_BRUSH = 2, 64
SAVE_DIR = "saved_numbers"        # Folder for images + labels.csv
FILE_PREFIX = "number_"           # number_1.png, number_2.png, ...
LABELS_CSV = "labels.csv"         # saved_numbers/labels.csv

# ---------- Helpers ----------
def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def next_index():
    """Find the next integer index by scanning existing 'number_*.png' files."""
    ensure_save_dir()
    indices = []
    pat = re.compile(rf"^{re.escape(FILE_PREFIX)}(\d+)\.png$")
    for fn in os.listdir(SAVE_DIR):
        m = pat.match(fn)
        if m:
            try:
                indices.append(int(m.group(1)))
            except ValueError:
                pass
    return (max(indices) + 1) if indices else 1

def save_sample(surface, label_text):
    """Save the drawing area (without UI strip) and append label to CSV."""
    idx = next_index()
    filename = f"{FILE_PREFIX}{idx}.png"
    path = os.path.join(SAVE_DIR, filename)

    # Crop out the UI area at the bottom
    rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
    cropped = surface.subsurface(rect).copy()
    pygame.image.save(cropped, path)

    # Append to CSV
    csv_path = os.path.join(SAVE_DIR, LABELS_CSV)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["filename", "label"])
        writer.writerow([filename, label_text])

    return idx, path

def draw_instructions(screen, font, label_text, brush_size):
    info_bg = (245, 245, 245)
    pygame.draw.rect(screen, info_bg, (0, HEIGHT, WIDTH, UI_HEIGHT))

    lines = [
        "Left-draw | Right-erase | C=clear | +/- change brush | Space=save | Esc=quit",
        f"Label: {label_text}",
        f"Brush: {brush_size}px   Saves to: ./{SAVE_DIR} (images) + labels.csv",
    ]
    y = HEIGHT + 8
    for line in lines:
        surf = font.render(line, True, (20, 20, 20))
        screen.blit(surf, (10, y))
        y += surf.get_height() + 6

def draw_line_with_brush(surf, color, start, end, radius):
    # Draw a thick line by interpolating points and stamping circles for smoothness
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

def main():
    pygame.init()
    pygame.display.set_caption("Number Drawer & Label Saver")
    screen = pygame.display.set_mode((WIDTH, HEIGHT + UI_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    big_font = pygame.font.SysFont(None, 28)

    # Main drawing surface (we'll draw UI onto same screen)
    screen.fill(BG_COLOR)

    brush_size = START_BRUSH
    drawing = False
    erasing = False
    last_pos = None
    label_text = ""

    ensure_save_dir()

    running = True
    while running:
        clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse input
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y < HEIGHT:  # ignore clicks in the UI area
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

            # Keyboard input
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    # Clear the drawing area (not the UI)
                    pygame.draw.rect(screen, BG_COLOR, (0, 0, WIDTH, HEIGHT))
                elif event.key == pygame.K_SPACE:
                    # Save image and label
                    idx, path = save_sample(screen, label_text.strip())
                    # Small visual toast
                    toast = big_font.render(f"Saved as {os.path.basename(path)}", True, (0, 120, 0))
                    pygame.draw.rect(screen, (235, 255, 235), (0, 0, WIDTH, toast.get_height() + 12))
                    screen.blit(toast, (10, 6))
                    # Reset drawing + label for next one (optional)
                    pygame.draw.rect(screen, BG_COLOR, (0, 0, WIDTH, HEIGHT))
                    label_text = ""
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    brush_size = min(MAX_BRUSH, brush_size + 1)
                elif event.key == pygame.K_MINUS:
                    brush_size = max(MIN_BRUSH, brush_size - 1)
                elif event.key == pygame.K_BACKSPACE:
                    if mods & (pygame.KMOD_CTRL | pygame.KMOD_META):
                        label_text = ""
                    else:
                        label_text = label_text[:-1]
                else:
                    # Accept printable characters for the label
                    ch = event.unicode
                    if ch and ch.isprintable():
                        label_text += ch

        # Redraw UI strip
        draw_instructions(screen, font, label_text, brush_size)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()

