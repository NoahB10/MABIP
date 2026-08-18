"""One-shot generator for the MABIP app icon (256x256 PNG).

Renders at 1024px and downsamples for clean anti-aliasing. Motif: a
96-well plate (8x12 dot grid) on a rounded-square background, with a
few wells "filled" in amber to suggest collected samples.

Run with the mabip env's python:
    /home/pi/miniconda3/envs/mabip/bin/python make_icon.py
"""
from PIL import Image, ImageDraw

S = 1024          # supersample canvas
OUT = 256         # final size

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Rounded-square background: deep teal with a subtle vertical gradient.
top = (16, 62, 84)      # deep teal
bot = (10, 36, 52)      # darker
radius = 180
grad = Image.new("RGBA", (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(grad)
for y in range(S):
    t = y / (S - 1)
    r = int(top[0] + (bot[0] - top[0]) * t)
    g = int(top[1] + (bot[1] - top[1]) * t)
    b = int(top[2] + (bot[2] - top[2]) * t)
    gd.line([(0, y), (S, y)], fill=(r, g, b, 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([8, 8, S - 8, S - 8], radius=radius, fill=255)
img.paste(grad, (0, 0), mask)

# Well-plate grid: 12 cols x 8 rows, like a 96-well plate.
cols, rows = 12, 8
margin_x, margin_y = 130, 240
cell_w = (S - 2 * margin_x) / (cols - 1)
cell_h = (S - margin_y - 170) / (rows - 1)
dot_r = 26
well = (137, 209, 224)          # pale cyan
amber = (245, 183, 77)          # sample-filled wells
filled = {(1, 2), (2, 5), (4, 8), (6, 3)}   # (row, col) accents
for r in range(rows):
    for c in range(cols):
        cx = margin_x + c * cell_w
        cy = margin_y + r * cell_h
        color = amber if (r, c) in filled else well
        d.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=color)

# Slim accent bar up top (nods to the flow/plot traces in the GUI).
d.rounded_rectangle([130, 110, 894, 160], radius=25, fill=(245, 183, 77))

img = img.resize((OUT, OUT), Image.LANCZOS)
img.save("/home/pi/Documents/MABIP-testing/packaging/mabip.png")
print("wrote mabip.png")
