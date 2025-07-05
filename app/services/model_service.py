import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from app.services.face_detector import detect_face_from_image  # sesuaikan path ini dengan struktur proyekmu

MODEL_PATH = 'model/100_do03_lr14.pth'

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['ASD', 'Normal']
    num_classes = len(class_names)

    model = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=False)
    model.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.head.in_features, 256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names

def predict_from_file(image_file, model, class_names):
    face_pil = detect_face_from_image(image_file)

    if face_pil is None:
        return {
            'error': 'Tidak ada wajah terdeteksi'
        }

    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    input_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze().tolist()

        # Ambil probabilitas tertinggi
    top_index = max(range(len(probs)), key=lambda i: probs[i])
    top_label = class_names[top_index]
    top_prob = round(probs[top_index], 4)

    # Hindari menampilkan 1.0 sebagai probabilitas
    if top_prob == 1.0:
        top_prob = 0.9999

    return {
        'result': top_label,
        'probability': top_prob
    }
