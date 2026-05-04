import json
import torch
import streamlit as st

from PIL import Image
from torch import nn
from torchvision import transforms, models


MODEL_PATH = "models/resnet18_pretrained_finetune.pth"
CLASS_PATH = "models/classes.json"


device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


st.title("Pokemon Classifier")
st.write("Upload a Pokemon image and the model will predict its name.")

uploaded_file = st.file_uploader("Choose a Pokemon image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    model, class_names = load_model()

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, 5)

    st.subheader("Top-5 Predictions")

    for i in range(5):
        class_name = class_names[top_idxs[0][i].item()]
        prob = top_probs[0][i].item() * 100
        st.write(f"{i+1}. {class_name}: {prob:.2f}%")