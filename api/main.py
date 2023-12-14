from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException

import pickle
from io import BytesIO
from PIL import Image
from .utils import cat_model, cat_predict, color_classifier, color_predict
from .utils import cat_model, cat_predict
import torch
from torchvision import models

app = FastAPI(
    title="ML fashion task",
)

# loading model wights
cat_model.load_state_dict(torch.load("models/custom_model.pth", map_location=torch.device('cpu')))
cat_model.eval()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

color_classifier.load_state_dict(torch.load("models/mc_model.pth", map_location=torch.device('cpu')))
color_classifier.eval()

# prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # reading file
    file_contents = await file.read()
    content_type = file.content_type

    # endpoint only accepts jpg files
    if content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    pil_image = Image.open(BytesIO(file_contents))
    
    # getting category and color predictions
    cat_pred = cat_predict(pil_image)
    rgb_vals, color_pred = color_predict(pil_image)

    return {"category": cat_pred,
            "color rgb value": rgb_vals,
            "color category": color_pred}