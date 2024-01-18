# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from src.models.transformer import *  # Assuming you have a ViT model
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

class TrainingConfig(BaseModel):
    # Define Pydantic models for the training configuration
    lr: float
    epochs: int
    ckpt_name: str

# Load the pre-trained model
model = ViT
checkpoint = torch.load('models/run_3/ckpt_1.pth')
print(checkpoint.keys())

model.load_state_dict(torch.load("models/run_3/ckpt_1.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((160, 106)),  # Adjust size based on your model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded image
    with open("temp.jpg", "wb") as buffer:
        buffer.write(file.file.read())

    # Open and preprocess the image
    image = Image.open("temp.jpg")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Process the output as needed
    result = {"prediction": output[0].item()}  # Adjust based on your model's output

    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)