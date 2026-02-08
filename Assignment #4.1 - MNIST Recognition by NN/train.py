"""
train.py - Train MLP, CNN, and Transformer models on MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

print("=" * 70)
print("MNIST TRAINING - MLP, CNN, and Transformer")
print("=" * 70)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

    def get_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN(nn.Module):
    """Convolutional Neural Network"""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pooling layers: 28→14→7→3
        self.flatten_size = 128 * 3 * 3

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, self.flatten_size)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def get_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Transformer(nn.Module):
    """Transformer for MNIST"""

    def __init__(self):
        super(Transformer, self).__init__()

        # Split image into 7x7 patches (4 patches from 28x28)
        self.patch_size = 7
        self.num_patches = (28 // 7) ** 2  # 16 patches
        patch_dim = 1 * 7 * 7  # 49

        self.patch_embed = nn.Linear(patch_dim, 128)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, 128))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        # Reshape to patches
        b, c, h, w = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, self.num_patches, -1)

        # Embed patches
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Class token
        x = self.norm(x)
        x = x[:, 0]

        # Classification
        x = self.head(x)

        return x

    def get_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# DATA LOADING
# ============================================================================

def get_mnist_data(batch_size=64, augment=True, val_split=0.1):
    """Get MNIST data loaders with train/validation split"""

    # Base transform
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Training transform with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = base_transform

    # Load full training dataset
    full_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    # Load test dataset (no augmentation)
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=base_transform
    )

    # Split into train/validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_single_model(model, model_name, train_loader, val_loader, device, epochs=10):
    """Train a single model and return training history"""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\nTraining {model_name}...")
    print(f"Parameters: {model.get_params():,}")
    print(f"Epochs: {epochs}")
    print("-" * 50)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, pred = output.max(1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, f'best_{model_name}.pth')

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Save final model
    torch.save(model.state_dict(), f'{model_name}_model.pth')

    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs, model_name)

    return model, train_losses, train_accs, val_losses, val_accs, best_val_acc


def plot_training_history(train_losses, train_accs, val_losses, val_accs, model_name):
    """Plot and save training history"""
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(val_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy History')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training.png', dpi=100)
    plt.close(fig)  # Close plot to prevent display


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main function to train all models"""

    # Create directories
    os.makedirs('results', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data
    print("\nLoading MNIST data with augmentation...")
    train_loader, val_loader, test_loader = get_mnist_data(
        batch_size=64,
        augment=True,
        val_split=0.1
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Define models to train
    models_config = [
        ('MLP', MLP()),
        ('CNN', CNN()),
        ('Transformer', Transformer())
    ]

    training_results = {}

    # Train each model
    for model_name, model in models_config:
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {model_name}")
        print(f"{'=' * 60}")

        # Move model to device
        model = model.to(device)

        # Train the model
        start_time = time.time()
        trained_model, train_losses, train_accs, val_losses, val_accs, best_val_acc = train_single_model(
            model, model_name, train_loader, val_loader, device, epochs=10
        )
        training_time = time.time() - start_time

        # Store results
        training_results[model_name] = {
            'model': trained_model,
            'train_acc': train_accs[-1],
            'val_acc': best_val_acc,
            'training_time': training_time,
            'parameters': model.get_params()
        }

        print(f"\n{model_name} Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Training Time: {training_time:.1f} seconds")
        print(f"Model saved as: {model_name}_model.pth")

    # Save training summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")

    with open('results/training_summary.txt', 'w') as f:
        f.write("MNIST Training Summary\n")
        f.write("=" * 50 + "\n\n")

        for model_name, results in training_results.items():
            print(f"{model_name:12s}: "
                  f"Val Acc: {results['val_acc']:.2f}%, "
                  f"Time: {results['training_time']:.1f}s, "
                  f"Params: {results['parameters']:,}")

            f.write(f"{model_name}:\n")
            f.write(f"  Validation Accuracy: {results['val_acc']:.2f}%\n")
            f.write(f"  Training Time: {results['training_time']:.1f} seconds\n")
            f.write(f"  Parameters: {results['parameters']:,}\n\n")

    print(f"\nAll models trained successfully!")
    print(f"Training plots saved to 'results/' folder")
    print(f"Models saved as: MLP_model.pth, CNN_model.pth, Transformer_model.pth")
    print(f"\nNow run: python test.py")


if __name__ == '__main__':
    main()