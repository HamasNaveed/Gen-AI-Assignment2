"""Flask demo for the CycleGAN face <-> sketch model.

Run:  python app.py        (weights expected in app/weights/G_face2sketch.pth and G_sketch2face.pth)
"""
import base64
import io
import os

import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, jsonify, render_template, request
from PIL import Image

from models import IMG_SIZE, Generator

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_models = {}


def load_generator(name):
    """Lazy-load and cache a generator ('face2sketch' or 'sketch2face')."""
    if name not in _models:
        path = os.path.join(WEIGHT_DIR, f"G_{name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weights: {path}. Copy them from Colab (see README).")
        g = Generator().to(DEVICE)
        g.load_state_dict(torch.load(path, map_location=DEVICE))
        _models[name] = g.eval()
    return _models[name]


def looks_like_sketch(img):
    """Heuristic: sketches are nearly colourless (low HSV saturation) and mostly bright paper-white."""
    hsv = np.asarray(img.resize((128, 128)).convert("HSV"), dtype=np.float32) / 255.0
    sat, val = hsv[..., 1].mean(), np.median(hsv[..., 2])
    return bool(sat < 0.10 and val > 0.65)


def translate(img, mode):
    """Run the right generator on a PIL image; returns (PIL output, detected_kind)."""
    kind = mode if mode in ("sketch", "face") else ("sketch" if looks_like_sketch(img) else "face")
    g = load_generator("sketch2face" if kind == "sketch" else "face2sketch")
    tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC), T.ToTensor(),
                    T.Normalize((.5, .5, .5), (.5, .5, .5))])
    with torch.no_grad():
        out = g(tf(img).unsqueeze(0).to(DEVICE))[0].cpu() * 0.5 + 0.5
    return T.ToPILImage()(out.clamp(0, 1)), kind


def to_data_url(img):
    """Encode a PIL image as a PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    """Accept an uploaded/captured image (field 'image') and optional 'mode' (auto|face|sketch)."""
    if "image" not in request.files:
        return jsonify(error="No image uploaded"), 400
    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
        out, kind = translate(img, request.form.get("mode", "auto"))
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except Exception as e:  # unreadable image etc.
        return jsonify(error=f"Could not process image: {e}"), 400
    return jsonify(detected=kind, result=to_data_url(out))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
