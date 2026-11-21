"""Procedural renderer for German traffic signs (GTSRB classes), no icon files needed.

Each sign is drawn from geometric primitives at 128 px, then pushed through a camera-like degradation
chain (perspective, rotation, natural background, lighting, blur, down-sampling, JPEG, noise) so that a
classifier trained on it has a chance to transfer to real 30 to 60 px road-side crops.

Only classes whose pictogram is simple geometry are rendered (see DRAWABLE). The renderer is used for
two things in experiments/gtsrb_unseen.py: classes that get no real training images at all, and an
ablation where seen classes also get synthetic copies, so 'looks synthetic' cannot become a shortcut
for 'is one of the unseen classes'.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RED = (200, 25, 30)
BLUE = (25, 85, 180)
YELLOW = (250, 200, 20)
WHITE = (245, 245, 240)
BLACK = (20, 20, 20)
GREY = (110, 110, 110)

SPEED = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
DRAWABLE = sorted([*SPEED, 6, 12, 13, 14, 15, 17, 18, 32, 33, 34, 35, 36, 37, 38, 39, 40])

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
]


def _font(size: int):
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _text(d: ImageDraw.ImageDraw, cx: float, cy: float, s: str, size: int, fill, squeeze: float = 1.0):
    f = _font(size)
    l, t, r, b = d.textbbox((0, 0), s, font=f)
    w, h = r - l, b - t
    if squeeze < 1.0:  # narrow the digits like the German road-sign typeface
        tmp = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((2 - l, 2 - t), s, font=f, fill=fill)
        tmp = tmp.resize((max(1, int((w + 4) * squeeze)), h + 4), Image.LANCZOS)
        d._image.paste(tmp, (int(cx - tmp.width / 2), int(cy - tmp.height / 2)), tmp)  # noqa: SLF001
    else:
        d.text((cx - w / 2 - l, cy - h / 2 - t), s, font=f, fill=fill)


def _arrow(d, pts, width, head, fill):
    """Polyline with a triangular head at the last point."""
    d.line(pts, fill=fill, width=width, joint="curve")
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    tip = (x1 + math.cos(a) * head * 0.9, y1 + math.sin(a) * head * 0.9)
    left = (x1 + math.cos(a + 2.3) * head, y1 + math.sin(a + 2.3) * head)
    right = (x1 + math.cos(a - 2.3) * head, y1 + math.sin(a - 2.3) * head)
    d.polygon([tip, left, right], fill=fill)


def draw_sign(cls: int, size: int = 128, rng: np.random.Generator | None = None) -> Image.Image:
    """RGBA image of one sign, transparent outside its shape."""
    rng = rng or np.random.default_rng()
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size / 2
    R = size * 0.47
    ring = R * 0.18
    w = max(2, int(size * 0.07))

    def circle(r, fill):
        d.ellipse([c - r, c - r, c + r, c + r], fill=fill)

    if cls in SPEED or cls in (15, 6, 32):  # white disc
        circle(R, RED if cls in SPEED or cls == 15 else BLACK if cls == 32 else GREY)
        circle(R - (ring if cls in SPEED or cls == 15 else R * 0.04), WHITE)
        if cls in SPEED:
            txt = SPEED[cls]
            _text(d, c, c, txt, int(size * (0.42 if len(txt) == 2 else 0.33)), BLACK, squeeze=0.8)
        if cls == 6:
            _text(d, c, c, "80", int(size * 0.42), GREY, squeeze=0.8)
            for k in (-1, 0, 1):
                o = k * size * 0.06
                d.line([(c + R * 0.62 + o, c - R * 0.62 + o), (c - R * 0.62 + o, c + R * 0.62 + o)], fill=GREY, width=max(1, w // 3))
        if cls == 32:
            for k in range(-2, 3):
                o = k * size * 0.05
                d.line([(c + R * 0.66 + o, c - R * 0.66 + o), (c - R * 0.66 + o, c + R * 0.66 + o)], fill=BLACK, width=max(1, w // 3))
    elif cls == 17:  # no entry
        circle(R, RED)
        d.rectangle([c - R * 0.68, c - R * 0.17, c + R * 0.68, c + R * 0.17], fill=WHITE)
    elif cls == 14:  # stop
        pts = [(c + R * math.cos(math.radians(22.5 + 45 * k)), c + R * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
        d.polygon(pts, fill=WHITE)
        pts2 = [(c + R * 0.9 * math.cos(math.radians(22.5 + 45 * k)), c + R * 0.9 * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
        d.polygon(pts2, fill=RED)
        _text(d, c, c, "STOP", int(size * 0.26), WHITE, squeeze=0.9)
    elif cls in (13, 18):  # triangles: yield points down, caution points up
        up = cls == 18
        h = R * 1.75
        top = c - h / 2 if up else c + h / 2
        base = c + h / 2 if up else c - h / 2
        outer = [(c, top), (c - R * 1.0, base), (c + R * 1.0, base)]
        d.polygon(outer, fill=RED)
        k = 0.62
        cy = c + (h * 0.12 if up else -h * 0.12)
        inner = [(c + (x - c) * k, cy + (y - c) * k) for x, y in outer]
        d.polygon(inner, fill=WHITE)
        if up:
            d.rectangle([c - size * 0.03, c - size * 0.08, c + size * 0.03, c + size * 0.12], fill=BLACK)
            d.ellipse([c - size * 0.035, c + size * 0.16, c + size * 0.035, c + size * 0.23], fill=BLACK)
    elif cls == 12:  # priority road
        q = R
        d.polygon([(c, c - q), (c + q, c), (c, c + q), (c - q, c)], fill=BLACK)
        q2 = R * 0.93
        d.polygon([(c, c - q2), (c + q2, c), (c, c + q2), (c - q2, c)], fill=WHITE)
        q3 = R * 0.62
        d.polygon([(c, c - q3), (c + q3, c), (c, c + q3), (c - q3, c)], fill=YELLOW)
    else:  # blue mandatory signs 33..40
        circle(R, WHITE)
        circle(R * 0.94, BLUE)
        aw = max(3, int(size * 0.09))
        hd = size * 0.16
        if cls == 35:
            _arrow(d, [(c, c + R * 0.6), (c, c - R * 0.35)], aw, hd, WHITE)
        elif cls in (33, 34):
            sgn = 1 if cls == 33 else -1
            _arrow(d, [(c - sgn * R * 0.2, c + R * 0.6), (c - sgn * R * 0.2, c - R * 0.05), (c + sgn * R * 0.35, c - R * 0.05)], aw, hd, WHITE)
        elif cls in (36, 37):
            sgn = 1 if cls == 36 else -1
            _arrow(d, [(c - sgn * R * 0.15, c + R * 0.65), (c - sgn * R * 0.15, c - R * 0.35)], aw, hd, WHITE)
            _arrow(d, [(c - sgn * R * 0.15, c + R * 0.2), (c + sgn * R * 0.4, c - R * 0.05)], aw, hd * 0.9, WHITE)
        elif cls in (38, 39):
            sgn = 1 if cls == 38 else -1
            _arrow(d, [(c - sgn * R * 0.35, c - R * 0.4), (c + sgn * R * 0.25, c + R * 0.25)], aw, hd, WHITE)
        elif cls == 40:
            for k in range(3):
                a0 = math.radians(120 * k - 90)
                pts = [(c + R * 0.5 * math.cos(a0 + t), c + R * 0.5 * math.sin(a0 + t)) for t in np.linspace(0, 1.4, 8)]
                _arrow(d, pts, max(2, aw - 2), hd * 0.7, WHITE)
        else:
            raise ValueError(f"class {cls} is not drawable")
    return im


def _perspective(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-strength, strength, size=(4, 2)) * np.array([w, h])
    M = cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32))
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def render_sample(cls: int, out: int, rng: np.random.Generator, backgrounds: np.ndarray | None = None) -> np.ndarray:
    """One degraded RGB sample of class `cls`, framed like a GTSRB ROI crop with ~10% margin."""
    import cv2

    S = 128
    sign = draw_sign(cls, S, rng)
    sign = sign.rotate(float(rng.uniform(-10, 10)), resample=Image.BILINEAR)
    rgba = _perspective(np.asarray(sign), rng, 0.08)
    # background: a random patch of a natural photo, or flat noise if none given
    canvas_size = int(S * rng.uniform(1.08, 1.3))
    if backgrounds is not None and len(backgrounds):
        bg = backgrounds[rng.integers(len(backgrounds))]
        y0, x0 = rng.integers(0, bg.shape[0] - 64, size=2)
        bg = cv2.resize(bg[y0: y0 + 64, x0: x0 + 64], (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)
        bg = cv2.GaussianBlur(bg, (0, 0), float(rng.uniform(0.5, 3)))
    else:
        bg = np.full((canvas_size, canvas_size, 3), rng.integers(40, 200, 3), np.uint8)
    canvas = bg.astype(np.float32)
    off = (canvas_size - S) // 2 + rng.integers(-4, 5, size=2)
    off = np.clip(off, 0, canvas_size - S)
    region = canvas[off[1]: off[1] + S, off[0]: off[0] + S]
    a = rgba[..., 3:4].astype(np.float32) / 255
    region[:] = region * (1 - a) + rgba[..., :3].astype(np.float32) * a
    # lighting: exposure, contrast, colour cast, gamma
    img = canvas * rng.uniform(0.35, 1.25) + rng.uniform(-25, 25)
    img = (img - img.mean()) * rng.uniform(0.6, 1.2) + img.mean()
    img *= rng.uniform(0.85, 1.15, size=3)
    img = 255 * (np.clip(img, 0, 255) / 255) ** rng.uniform(0.7, 1.5)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # optics and sensor: blur, motion blur, low resolution, JPEG, noise
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 2.0)))
    if rng.random() < 0.3:
        k = int(rng.integers(3, 9))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), float(rng.uniform(0, 180)), 1.0)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kern, M, (k, k)))
    low = int(rng.uniform(18, 56))
    img = cv2.resize(cv2.resize(img, (low, low), interpolation=cv2.INTER_AREA), (out, out), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.7:
        ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 90))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    img = img.astype(np.float32) + rng.normal(0, rng.uniform(0, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def contact_sheet(classes, out: int = 64, per_class: int = 6, seed: int = 0, backgrounds=None) -> Image.Image:
    rng = np.random.default_rng(seed)
    sheet = Image.new("RGB", (out * per_class, out * len(classes)), (255, 255, 255))
    for r, cls in enumerate(classes):
        for k in range(per_class):
            sheet.paste(Image.fromarray(render_sample(cls, out, rng, backgrounds)), (k * out, r * out))
    return sheet


if __name__ == "__main__":
    import sys

    dst = Path(sys.argv[1] if len(sys.argv) > 1 else "synth_preview.png")
    clean = Image.new("RGB", (96 * len(DRAWABLE), 96), (255, 255, 255))
    for i, c in enumerate(DRAWABLE):
        s = draw_sign(c, 96)
        clean.paste(s, (i * 96, 0), s)
    clean.save(dst.with_name(dst.stem + "_clean.png"))
    contact_sheet(DRAWABLE[:12]).save(dst)
    print("wrote", dst)
