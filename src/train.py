import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader, len(dataset.classes)


def build_model(num_classes):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train(model, dataloader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple dog/cat/other classifier")
    parser.add_argument("data_dir", help="Path to dataset with subfolders for each class")
    parser.add_argument("output", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    loader, num_classes = load_data(args.data_dir, args.batch_size)
    model = build_model(num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(model, loader, device, args.epochs, args.lr)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")
