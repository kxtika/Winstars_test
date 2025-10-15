
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from .mnist_interface import MnistClassifierInterface

class FeedForwardNet(nn.Module):
    def __init__(self, in_features: int = 28*28, hidden: Tuple[int, int] = (256, 128), num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden[1], num_classes)
        )

    def forward(self, x):
        return self.net(x)

class FeedForwardMnistClassifier(MnistClassifierInterface):
    def __init__(self, lr: float = 1e-3, epochs: int = 5, device: str = None):
        self.lr = lr
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FeedForwardNet().to(self.device)

    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        # X_train: torch Dataset OR Tensor of shape [N,1,28,28]
        # y_train: Tensor [N]
        train_loader = self._as_loader(X_train, y_train, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            print(f"[FFNN] Epoch {epoch+1}/{self.epochs} loss={total_loss/len(train_loader.dataset):.4f}")

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
                pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu())
        return torch.cat(preds, dim=0)

    @staticmethod
    def _as_loader(X, y=None, batch_size=128, shuffle=False, y_required=True):
        import torch
        from torch.utils.data import TensorDataset, DataLoader, Dataset
        if isinstance(X, Dataset):
            ds = X
        elif isinstance(X, torch.Tensor):
            if y_required and y is None:
                raise ValueError("y must be provided when X is tensor")
            ds = TensorDataset(X.float(), y.long()) if y_required else TensorDataset(X.float())
        else:
            raise TypeError("Unsupported X type; provide a PyTorch Dataset or Tensor")
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
