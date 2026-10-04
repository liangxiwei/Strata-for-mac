#!/usr/bin/env python3
"""The test pictures (images/): drawn here so their content is known exactly - text to read, shapes and colours to
name, dots to count - plus the repository's own screenshot of the Strata app, scaled down."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)
ROOT = Path(__file__).resolve().parents[3]


def font(size, names=("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc")):
    for n in names:
        if Path(n).exists():
            return ImageFont.truetype(n, size)
    return ImageFont.load_default()


# 1. text: a receipt-like card (read the words and the number)
im = Image.new("RGB", (800, 500), "white")
d = ImageDraw.Draw(im)
d.text((60, 60), "STRATA COFFEE", fill="black", font=font(64))
d.text((60, 180), "Latte  x2      9.40", fill="black", font=font(44))
d.text((60, 250), "Bagel  x1      3.20", fill="black", font=font(44))
d.text((60, 360), "TOTAL         12.60", fill="black", font=font(52))
im.save(OUT / "receipt.png")

# 2. shapes: a red circle, a blue square, a green triangle (name each one's colour)
im = Image.new("RGB", (900, 360), "white")
d = ImageDraw.Draw(im)
d.ellipse((60, 60, 300, 300), fill=(220, 30, 30))
d.rectangle((340, 60, 580, 300), fill=(30, 60, 220))
d.polygon([(740, 50), (620, 310), (860, 310)], fill=(30, 170, 60))
im.save(OUT / "shapes.png")

# 3. counting: seven orange dots on grey
im = Image.new("RGB", (800, 400), (235, 235, 235))
d = ImageDraw.Draw(im)
for i, (x, y) in enumerate([(100, 100), (260, 140), (420, 90), (580, 150), (180, 280), (380, 300), (620, 290)]):
    d.ellipse((x - 40, y - 40, x + 40, y + 40), fill=(245, 140, 20))
im.save(OUT / "dots.png")

# 4. Chinese text (read it back)
cjk = font(72, ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))
im = Image.new("RGB", (800, 300), "white")
ImageDraw.Draw(im).text((60, 100), "苹果电脑运行大模型", fill="black", font=cjk)
im.save(OUT / "chinese.png")

# 5. a real screenshot: the Strata app's Monitor next to a coding agent (docs/media/runpagoda.png), as a JPEG
src = Image.open(ROOT / "docs" / "media" / "runpagoda.png").convert("RGB")
src.thumbnail((1280, 1280))
src.save(OUT / "screenshot.jpg", quality=88)
print("\n".join(f"{p.name} {Image.open(p).size}" for p in sorted(OUT.iterdir())))
