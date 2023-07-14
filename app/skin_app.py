import os, json, pickle, warnings
import base64, io, pathlib

import numpy as np
import torch
from flask_cors import CORS
from flask import Flask, request, Response
from PIL import Image

from skin_net.skin_net import SkinNet
from skin_net.skin_util import DATA_TRANSFORMS


app = Flask(__name__)
CORS(app, support_credentials=True)
MODEL_PATH = pathlib.Path(__file__).parent/"models/attempt-5-best.pth"

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_MODEL = SkinNet(num_class=2, pretrained=False).to(DEVICE)
IMAGE_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
IMAGE_MODEL.eval()


@app.route("/detect_skin_defect", methods=["POST"])
def skin_defects_from_image():
    class_to_idx = {
                "Atopic Eczema":0, "Non Atopic Eczema":1
            }
    content_type = str(request.content_type).split(';')[0]
    try:
        if content_type == "application/json":
            b64_encoded_img = request.get_json()["image"]
            image = Image.open(io.BytesIO(base64.b64decode(b64_encoded_img))).convert("RGB")
            image_t = DATA_TRANSFORMS["test"](image).unsqueeze(0).to(DEVICE)
            output = IMAGE_MODEL(image_t)
            output_probs = torch.nn.functional.softmax(output, dim=1)
            idx_to_class = {V:K for K, V in class_to_idx.items()}
            probs, classes = torch.topk(output_probs, 2)
            classes = [idx_to_class[i] for i in classes.squeeze().tolist()]
        
            return Response(
                        response=json.dumps(
                                {
                                    "predicted_classes": classes,
                                    "probabilities":probs.squeeze().tolist(),
                                    "warning": ""
                                }
                            ),
                        status=200,
                        mimetype="application/json"
                    )

        raise ValueError("Invalid Content-Type") from None
        
    except Exception as e:
        print(e)
        return Response(
                    response=json.dumps(
                        {
                            "warning" : "An error occured while processing request. "\
                                        "Ensure request's json body is in right format"\
                                        " and Content-Type is application/json.",
                            "predicted_classes":[],
                            "probabilities":[],
                        }
                    ),
                    status=400,
                    mimetype="application/json"
                )


if __name__ == "__main__":
    app.run(debug=True)
