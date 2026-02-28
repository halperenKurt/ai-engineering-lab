import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def run_experiment(
    train_set: datasets.MNIST,
    test_set: datasets.MNIST,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history.append(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )

    final = history[-1]
    return {
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "final_train_loss": final["train_loss"],
        "final_train_accuracy": final["train_accuracy"],
        "final_test_loss": final["test_loss"],
        "final_test_accuracy": final["test_accuracy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="../day-001-mnist-cnn/data")
    parser.add_argument("--output", type=str, default="sweep_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    data_dir = Path(args.data_dir)
    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    search_space = [
        {"learning_rate": 1e-3, "batch_size": 64},
        {"learning_rate": 1e-3, "batch_size": 128},
        {"learning_rate": 5e-4, "batch_size": 64},
        {"learning_rate": 5e-4, "batch_size": 128},
    ]

    results = []
    for idx, params in enumerate(search_space, start=1):
        print(
            f"[{idx}/{len(search_space)}] "
            f"lr={params['learning_rate']} batch_size={params['batch_size']}"
        )
        exp_result = run_experiment(
            train_set=train_set,
            test_set=test_set,
            lr=params["learning_rate"],
            batch_size=params["batch_size"],
            epochs=args.epochs,
            seed=args.seed,
            device=device,
        )
        print(
            f"  -> test_acc={exp_result['final_test_accuracy']:.4f} "
            f"test_loss={exp_result['final_test_loss']:.4f}"
        )
        results.append(exp_result)

    best = max(results, key=lambda x: x["final_test_accuracy"])
    output_payload = {"device": str(device), "results": results, "best": best}

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved sweep results to {output_path.resolve()}")
    print(
        f"Best config: lr={best['learning_rate']} batch_size={best['batch_size']} "
        f"test_acc={best['final_test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()

