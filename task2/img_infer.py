
import argparse, json, torch, os
from PIL import Image
from torchvision import transforms, models
from torch import nn

def load_model(model_path, class_names_json=None, device=None):
    ckpt = torch.load(model_path, map_location='cpu')
    arch = ckpt.get('arch', 'resnet18')
    classes = ckpt.get('classes')
    if class_names_json and os.path.exists(class_names_json):
        with open(class_names_json) as f:
            classes = json.load(f)
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, len(classes))
    else:
        model = models.mobilenet_v3_small(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, len(classes))
    model.load_state_dict(ckpt['state_dict'])
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, classes, device

def predict_image(model, device, image_path):
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    return logits.softmax(1).squeeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--image_path', required=True)
    ap.add_argument('--class_names_json', default=None)
    args = ap.parse_args()
    model, classes, device = load_model(args.model_path, args.class_names_json)
    probs = predict_image(model, device, args.image_path)
    idx = int(probs.argmax().item())
    print({'label': classes[idx], 'prob': float(probs[idx])})

if __name__ == '__main__':
    main()
