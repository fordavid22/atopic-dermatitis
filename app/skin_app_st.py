import os, json, pickle, warnings
import base64, io, pathlib

import numpy as np
import torch
from PIL import Image

from app.skin_net.skin_net import SkinNet
from app.skin_net.skin_util import DATA_TRANSFORMS


MODEL_PATH = pathlib.Path(__file__).parent/"models/attempt-8-best.pth"

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_MODEL = SkinNet(num_class=2, pretrained=False).to(DEVICE)
IMAGE_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
IMAGE_MODEL.eval()


def skin_defects_from_image(body):
    class_to_idx = {
                "Atopic Eczema":0, "Non Atopic Eczema":1
            }
    if body:
        b64_encoded_img = body["image"]
        image = Image.open(io.BytesIO(base64.b64decode(b64_encoded_img))).convert("RGB")
        image_t = DATA_TRANSFORMS["test"](image).unsqueeze(0).to(DEVICE)
        output = IMAGE_MODEL(image_t)
        output_probs = torch.nn.functional.softmax(output, dim=1)
        idx_to_class = {V:K for K, V in class_to_idx.items()}
        probs, classes = torch.topk(output_probs, 2)
        classes = [idx_to_class[i] for i in classes.squeeze().tolist()]
    
        return {
                "predicted_classes": classes,
                "probabilities":probs.squeeze().tolist()
            }

    else:
        return {}
