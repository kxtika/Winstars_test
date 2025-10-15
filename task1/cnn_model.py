
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .mnist_interface import MnistClassifierInterface

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ConvolutionalMnistClassifier(MnistClassifierInterface):
    def __init__(self, lr: float = 1e-3, epochs: int = 5, device: str = None):
        self.lr = lr
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN().to(self.device)

    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        loader = self._as_loader(X_train, y_train, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * xb.size(0)
            print(f"[CNN] Epoch {epoch+1}/{self.epochs} loss={total/len(loader.dataset):.4f}")

    def predict(self, X: Any) -> Any:
        loader = self._as_loader(X, None, batch_size=256, shuffle=False, y_required=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                xb = xb.to(self.device)
                logits = self.model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu())
        return torch.cat(preds, dim=0)

    @staticmethod
    def _as_loader(X, y=None, batch_size=128, shuffle=False, y_required=True):
        if isinstance(X, Dataset):
            ds = X
        elif isinstance(X, torch.Tensor):
            if y_required and y is None:
                raise ValueError("y must be provided when X is tensor")
            ds = TensorDataset(X.float(), y.long()) if y_required else TensorDataset(X.float())
        else:
            raise TypeError("Unsupported X type; provide a PyTorch Dataset or Tensor")
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
