
import argparse, os, json
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


def build_loaders(data_dir, batch_size=32):
    tf_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=tf_train)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=tf_val)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            train_ds.classes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='folder with train/ and val/ subfolders')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_path', default='checkpoints/img_model.pt')
    ap.add_argument('--arch', default='resnet18', choices=['resnet18','mobilenet_v3_small'])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, classes = build_loaders(args.data_dir, args.batch_size)

    if args.arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, len(classes))
    else:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, len(classes))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += float(loss) * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        print(f'Epoch {epoch+1}: train loss={loss_sum/total:.4f} acc={correct/total:.3f}')

        # quick val
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += xb.size(0)
        print(f'          val acc={correct/total:.3f}')

    torch.save({'state_dict': model.state_dict(), 'classes': classes, 'arch': args.arch}, args.out_path)
    with open(os.path.splitext(args.out_path)[0] + '_classes.json', 'w') as f:
        json.dump(classes, f)
    print('Saved to', args.out_path)

if __name__ == '__main__':
    main()
