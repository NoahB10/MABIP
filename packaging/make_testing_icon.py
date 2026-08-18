"""Amber variant of the app icon for the TESTING desktop entry, so the two
launchers are distinguishable at a glance. Same well-plate motif.

Run with the mabip env's python:
    /home/pi/miniconda3/envs/mabip/bin/python make_testing_icon.py
"""
from PIL import Image, ImageDraw

S, OUT = 1024, 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

top = (140, 84, 15)     # amber-brown
bot = (84, 48, 8)
grad = Image.new("RGBA", (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(grad)
for y in range(S):
    t = y / (S - 1)
    gd.line([(0, y), (S, y)],
            fill=(int(top[0] + (bot[0] - top[0]) * t),
                  int(top[1] + (bot[1] - top[1]) * t),
                  int(top[2] + (bot[2] - top[2]) * t), 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([8, 8, S - 8, S - 8], radius=180, fill=255)
img.paste(grad, (0, 0), mask)

cols, rows = 12, 8
margin_x, margin_y = 130, 240
cell_w = (S - 2 * margin_x) / (cols - 1)
cell_h = (S - margin_y - 170) / (rows - 1)
well = (247, 214, 160)
accent = (137, 209, 224)
filled = {(1, 2), (2, 5), (4, 8), (6, 3)}
for r in range(rows):
    for c in range(cols):
        cx, cy = margin_x + c * cell_w, margin_y + r * cell_h
        color = accent if (r, c) in filled else well
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=color)
d.rounded_rectangle([130, 110, 894, 160], radius=25, fill=(137, 209, 224))

img.resize((OUT, OUT), Image.LANCZOS).save(
    "/home/pi/Documents/MABIP-testing/packaging/mabip-testing.png")
print("wrote mabip-testing.png")
