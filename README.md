Winstars AI DS Internship Test

This repository contains structured, reproducible solutions for both tasks of the **Winstars AI Data Science Internship Test 2025**.  
It includes commented, ready-to-run Python scripts and per-task READMEs explaining data, training, and inference steps.

---

## Repository Structure
```
Winstars_DS_Test/
├── task1/                        # Task 1 – MNIST OOP classifiers
│   ├── mnist_classifier.py       # Unified OOP wrapper for multiple models
│   ├── mnist_train.ipynb         # Notebook version (EDA + training examples)
│   ├── data/                     # MNIST dataset (auto-downloaded)
│   ├── checkpoints/              # Trained model weights
│   ├── requirements.txt
│   └── README.md
│
└── task2/                        # Task 2 – NER + Animal Image Classification
    ├── data/
    │   ├── animals/              # 10-class image dataset (not included)
    │   └── ner/                  # NER CSV datasets (train/val)
    ├── checkpoints/              # Trained models
    ├── img_train.py              # CNN animal classifier training
    ├── ner_train.py              # DistilBERT NER fine-tuning
    ├── pipeline.py               # Combined multimodal inference
    ├── requirements.txt
    └── README.md
```

---

## Task 1 — MNIST OOP Classifiers

**Goal:** build and evaluate multiple digit classifiers (Random Forest, Feed-Forward NN, Convolutional NN) under a unified object-oriented API.

**Key Features**
- Unified `MnistClassifier` class handling data loading, training, and prediction.  
- Configurable model type: `rf`, `ffnn`, or `cnn`.  
- Example notebook with accuracy comparison and confusion matrices.

**Quick Start**
```bash
cd task1
pip install -r requirements.txt
python mnist_train.py             
```

**Example usage**
```python
from mnist_classifier import MnistClassifier

clf = MnistClassifier("cnn")
clf.train()
preds = clf.predict(test_images)
```

## Task 2 — NER + Animal Image Classification Pipeline

**Goal:** combine computer vision and NLP by linking text-extracted animal entities with predicted image classes.

**Components**
1. **Image Classifier:** CNN trained on 10 animal categories.  
2. **NER Model:** DistilBERT fine-tuned to tag animal names in text.  
3. **Pipeline:** verifies whether an image and a sentence refer to the same animal.

**Quick Start**
```bash
cd task2
pip install -r requirements.txt
python img_train.py --data_dir data/animals --epochs 5 --out_path checkpoints/img_model.pt
python ner_train.py --model_name distilbert-base-cased --train_csv data/ner/ner_train.csv --val_csv data/ner/ner_val.csv --out_dir checkpoints/ner
```

**Example Inference**
```bash
python pipeline.py --text "A dog is running in the field" --image data/animals/dog/example.jpg
```
Output:
```
NER found animals: ['dog']
Predicted label: dog
✅ MATCH between text and image!
```

---

## Global Requirements
Install dependencies inside a virtual environment:
```bash
pip install torch torchvision transformers datasets accelerate pandas numpy tqdm Pillow scikit-learn
```

Each task also provides its own `requirements.txt` for isolated setup.
