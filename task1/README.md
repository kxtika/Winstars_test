# Task 1 — MNIST Classification + OOP

Implements three MNIST classifiers hidden behind a common interface:
- Random Forest (scikit-learn) on flattened pixels
- Feed-Forward Neural Network (PyTorch)
- Convolutional Neural Network (PyTorch)

## Structure
```
task1/
  mnist_interface.py
  rf_model.py
  nn_model.py
  cnn_model.py
  mnist_classifier.py
  notebook.ipynb
  requirements.txt
  README.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run the demo notebook to see training & prediction examples.

## Notes
- RandomForest expects flattened inputs (N, 784). The NN/CNN accept tensors (N,1,28,28).
- `MnistClassifier` normalizes outputs to label vectors for consistency across algorithms.
