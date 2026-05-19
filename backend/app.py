from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
import cv2

app = FastAPI()

device = torch.device("cpu")

# ----------------------------
# LOAD FULL MODEL
# ----------------------------
MODEL_PATH = "/Users/jiheneguesmi/Desktop/machine_learning/smart_agriculture/notebooks/runs/full_model.pth"

model = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.to(device)
model.eval()

# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess(image_bytes):

    image = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (256, 256))

    image = image / 255.0

    image = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1)

    return image.unsqueeze(0)

# ----------------------------
# PREDICT
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = preprocess(image_bytes).to(device)

    with torch.no_grad():

        outputs = model(image)

        pred = outputs["out"]

        pred = torch.sigmoid(pred)

        pred = (pred > 0.5).float()

    mask = pred.squeeze().cpu().numpy()

    return {
        "mask": mask.tolist()
    }