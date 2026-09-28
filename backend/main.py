from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import io


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="CIFAR-10 ResNet18 API",
    description="CIFAR-10 image classification using a trained ResNet18 model",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# CIFAR-10 CLASS NAMES
# ============================================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# MODEL
# ============================================================

# Create ResNet18
resnet_model = models.resnet18(weights=None)

# CIFAR-10 = 10 classes
resnet_model.fc = nn.Linear(
    resnet_model.fc.in_features,
    10
)


# ============================================================
# MODEL PATH
# ============================================================

# Project structure:
#
# PYTORCH_PROJECT_06/
# │
# ├── backend/
# │   └── main.py
# │
# ├── model/
# │   └── Multi_class_Ditection.pth
# │
# └── frontend/
#

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = (
    BASE_DIR
    / "model"
    / "Multi_class_Ditection.pth"
)

print(f"Loading model from: {model_path}")


# Check if model exists
if not model_path.exists():

    raise FileNotFoundError(
        f"Model file not found:\n{model_path}"
    )


# Load model weights
resnet_model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)


# Move model to device
resnet_model = resnet_model.to(device)

# Evaluation mode
resnet_model.eval()

print("Model loaded successfully!")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "CIFAR-10 ResNet18 API is running",
        "model": "ResNet18",
        "dataset": "CIFAR-10",
        "classes": class_names,
        "device": str(device)
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check file type
    # --------------------------------------------------------

    if not file.content_type.startswith("image/"):

        return {
            "error": "Please upload a valid image file."
        }


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    image_tensor = transform(image)

    # Add batch dimension
    # [3, 224, 224]
    #       ↓
    # [1, 3, 224, 224]

    image_tensor = image_tensor.unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = resnet_model(
            image_tensor
        )

        # Convert logits → probabilities
        probabilities = torch.softmax(
            output,
            dim=1
        )

        # Highest probability class
        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        # Confidence
        confidence = probabilities[
            0,
            predicted_index
        ].item()


    # --------------------------------------------------------
    # Class name
    # --------------------------------------------------------

    predicted_class = class_names[
        predicted_index
    ]


    # --------------------------------------------------------
    # All class probabilities
    # --------------------------------------------------------

    all_probabilities = {
        class_names[i]: round(
            probabilities[0, i].item() * 100,
            2
        )
        for i in range(len(class_names))
    }


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "predicted_class": predicted_class,

        "class_index": predicted_index,

        "confidence": round(
            confidence * 100,
            2
        ),

        "probabilities": all_probabilities

    }