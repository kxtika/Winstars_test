
import argparse, json, os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def load_labels(model_dir):
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        m = json.load(f)
    return m['tag2id'], {int(k):v for k,v in m['id2tag'].items()}

def extract_animals(text, model, tokenizer, id2tag):
    tokens = text.split()
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    labels = logits.argmax(-1).tolist()
    word_ids = enc.word_ids(0)
    words = []
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        tag = id2tag.get(labels[i], 'O')
        if tag.startswith('B-ANIMAL') or tag == 'ANIMAL' or tag.endswith('ANIMAL'):
            words.append(tokens[wid])
    return list(dict.fromkeys(words))  # unique preserve order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--text', required=True)
    args = ap.parse_args()

    tag2id, id2tag = load_labels(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    animals = extract_animals(args.text, model, tokenizer, id2tag)
    print(animals)

if __name__ == '__main__':
    main()
