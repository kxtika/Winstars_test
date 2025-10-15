
import argparse, json, os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from PIL import Image
from torchvision import transforms, models
from torch import nn

def load_ner(model_dir):
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        m = json.load(f)
    id2tag = {int(k):v for k,v in m['id2tag'].items()}
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir)
    return tok, mdl, id2tag

def ner_animals(text, tokenizer, model, id2tag):
    tokens = text.split()
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    labels = logits.argmax(-1).tolist()
    word_ids = enc.word_ids(0)
    animals = []
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        tag = id2tag.get(labels[i], 'O')
        if 'ANIMAL' in tag:
            animals.append(tokens[wid].lower())
    return list(dict.fromkeys(animals))

def load_img_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    arch = ckpt.get('arch', 'resnet18')
    classes = [c.lower() for c in ckpt['classes']]
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, len(classes))
    else:
        model = models.mobilenet_v3_small(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, len(classes))
    model.load_state_dict(ckpt['state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, classes, device

def predict_label(model, device, image_path):
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    idx = int(logits.argmax(1).item())
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='NER model dir')
    ap.add_argument('--img_model', required=True, help='image classifier checkpoint .pt')
    ap.add_argument('--text', required=True)
    ap.add_argument('--image', required=True)
    args = ap.parse_args()

    tok, ner_m, id2tag = load_ner(args.model_dir)
    animals_text = ner_animals(args.text, tok, ner_m, id2tag)

    img_m, classes, device = load_img_model(args.img_model)
    img_idx = predict_label(img_m, device, args.image)
    img_label = classes[img_idx]

    decision = any(a == img_label for a in animals_text)
    print({'text_animals': animals_text, 'image_label': img_label, 'decision': bool(decision)})

if __name__ == '__main__':
    main()
