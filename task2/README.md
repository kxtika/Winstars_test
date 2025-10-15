Task 2  

This project demonstrates two complementary models:  
1.  A **CNN-based image classifier** that recognises 10 animal categories.  
2.  A **DistilBERT-based Named Entity Recognition (NER)** model that tags animal names in text.  

Together they form a small **text + image validation pipeline** similar to a multimodal reasoning system.

---

## 🗂 Folder Structure
```
Winstars_DS_Test/
└── task2/
    ├── data/
    │   ├── animals/               # 28 K animal images (10 classes) – not included
    │   └── ner/
    │       ├── ner_train.csv
    │       └── ner_val.csv
    │
    ├── checkpoints/
    │   ├── img_model.pt           # trained image model
    │   └── ner/                   # fine-tuned DistilBERT weights + tokenizer
    │
    ├── img_train.py               # image training script
    ├── ner_train.py               # NER fine-tuning script
    ├── pipeline.py                # combined text + image test pipeline
    ├── requirements.txt
    └── README.md
```
### Animal Image Dataset
The model expects a dataset structured as follows:
```
data/animals/
├── dog/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── cat/
├── horse/
├── cow/
├── elephant/
├── spider/
├── butterfly/
├── squirrel/
├── chicken/
└── sheep/
```

Each folder name represents the **class label**.

#### Option 1 — Manual Download (Recommended)
Download the 10-class “Animals” dataset (~28K images) from this link:  
🔗 **[Kaggle – Animals Dataset (10 Classes)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)**  
Unzip it and rename the extracted folder to `data/animals`.

#### Option 2 — Create Your Own
If Kaggle isn’t accessible, you can simulate a mini dataset by creating a few folders (dog, cat, etc.) and adding a handful of images to each (5–10 samples).  
The CNN will still train for demonstration purposes.

---

### NER Dataset
The NER training data is included in:
```
data/ner/
├── ner_train.csv
└── ner_val.csv
```

Each CSV has two columns:
```
tokens,tags
A cat is sleeping on the sofa,O B-ANIMAL O O O O O
...
```

If you want to expand or rebalance it:
- Add more **neutral sentences** (with tag `O` only).
- Keep animal names tagged as `B-ANIMAL`.
- Make sure each file follows the same token/tag structure.

### Train/Validation Split
Once downloaded, split your dataset into **training** and **validation** folders.  
For example:
```
data/animals/
├── train/
│   ├── dog/
│   ├── cat/
│   └── ...
└── val/
    ├── dog/
    ├── cat/
    └── ...
```
A common split ratio is **80% training / 20% validation**.  
You can do this manually or automatically with the following command-line utility:

```bash
pip install split-folders
python -m split_folders data/animals --output data/animals_split --ratio .8 .2
```

Then use:
```bash
python img_train.py --data_dir data/animals_split/train --epochs 5 --out_path checkpoints/img_model.pt
```
---

## 1. Image Classifier Training
Download or prepare the *10-class animal dataset* (dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant).

```bash
python img_train.py --data_dir data/animals --epochs 5 --out_path checkpoints/img_model.pt
```

Output example:
```
Epoch 1 – val_acc = 0.76  
Epoch 5 – val_acc ≈ 0.90  
Saved model to checkpoints/img_model.pt
```

---

## 2. NER Model Training
Train DistilBERT to recognise animal names in text.

```bash
python ner_train.py   --model_name distilbert-base-cased   --train_csv data/ner/ner_train.csv   --val_csv data/ner/ner_val.csv   --out_dir checkpoints/ner
```

Output example:
```
Training on cuda
Epoch 1 – loss = 0.31  val_acc = 0.93
Epoch 3 – loss = 0.12  val_acc = 0.97
Saved model to checkpoints/ner
```

---

## Testing Each Model

### NER inference
```bash
python - <<'PY'
from transformers import pipeline
ner = pipeline("token-classification",
               model=r"task2/checkpoints/ner",
               aggregation_strategy="simple",
               model_kwargs={"local_files_only": True})
print(ner("A dog and an elephant are walking in the field"))
PY
```

Example output (after adding id2label mapping in `config.json`):
```python
[
 {'entity_group': 'B-ANIMAL', 'word': 'dog'},
 {'entity_group': 'B-ANIMAL', 'word': 'elephant'}
]
```

### Image inference
```bash
python - <<'PY'
import torch
from torchvision import transforms
from PIL import Image

model = torch.load("task2/checkpoints/img_model.pt", map_location="cpu")
model.eval()

img = Image.open("task2/data/animals/dog/example.jpg")
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
x = transform(img).unsqueeze(0)
with torch.no_grad():
    pred = model(x).argmax(1).item()
print("Predicted class index:", pred)
PY
```

---

## Combined Pipeline
`pipeline.py` unites both models for text + image matching.

```bash
python task2/pipeline.py   --text "A cow and a butterfly are in the field"   --image task2/data/animals/dog/example.jpg
```

Example output:
```
NER found animals: ['cow', 'butterfly']
Predicted label: cow
MATCH between text and image!
```

---

## Dependencies
Install everything with:
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
torch>=2.0
torchvision>=0.15
transformers>=4.57
datasets
accelerate
pandas
numpy
tqdm
Pillow
scikit-learn
```
