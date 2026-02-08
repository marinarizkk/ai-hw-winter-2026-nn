"""
test.py - Test trained MLP, CNN, and Transformer models on MNIST
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("=" * 70)
print("MNIST TESTING - MLP, CNN, and Transformer")
print("=" * 70)


# ============================================================================
# MODEL DEFINITIONS (MUST MATCH train.py)
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


class SimpleTransformer(nn.Module):
    """Simplified Transformer for MNIST"""

    def __init__(self):
        super(SimpleTransformer, self).__init__()

        self.patch_size = 7
        self.num_patches = (28 // 7) ** 2
        patch_dim = 1 * 7 * 7

        self.patch_embed = nn.Linear(patch_dim, 128)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, 128))

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
        b, c, h, w = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, self.num_patches, -1)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x

    def get_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def load_test_data(batch_size=64):
    """Load MNIST test data"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,  # Already downloaded by train.py
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")
    return test_loader


def test_single_model(model_name, device):
    """Test a single trained model"""

    # Load the correct model architecture
    if model_name == 'MLP':
        model = MLP()
        model_file = 'MLP_model.pth'
    elif model_name == 'CNN':
        model = CNN()
        model_file = 'CNN_model.pth'
    elif model_name == 'Transformer':
        model = SimpleTransformer()
        model_file = 'Transformer_model.pth'
    else:
        print(f"Unknown model: {model_name}")
        return None

    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"✓ Loaded {model_name} from {model_file}")
    except FileNotFoundError:
        print(f"✗ ERROR: {model_file} not found. Run train.py first!")
        return None

    model = model.to(device)
    model.eval()

    return model


def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model on test set"""

    all_predictions = []
    all_targets = []

    print(f"\nTesting {model_name} on {len(test_loader.dataset)} images...")

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    correct = (all_predictions == all_targets).sum()
    total = len(all_targets)
    accuracy = 100.0 * correct / total

    # Per-class accuracy
    class_accuracies = []
    for digit in range(10):
        mask = all_targets == digit
        if mask.sum() > 0:
            class_acc = 100.0 * (all_predictions[mask] == digit).sum() / mask.sum()
            class_accuracies.append(class_acc)

    return accuracy, class_accuracies, all_predictions, all_targets


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion.png', dpi=100)
    plt.close()


def plot_sample_predictions(model, test_loader, device, model_name, num_samples=12):
    """Plot and save sample predictions"""
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images = images[:num_samples].to(device)
    labels = labels[:num_samples].cpu().numpy()

    with torch.no_grad():
        outputs = model(images)
        _, predictions = outputs.max(1)
        predictions = predictions.cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    for i in range(num_samples):
        img = images[i].cpu().numpy().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        color = 'green' if predictions[i] == labels[i] else 'red'
        axes[i].set_title(f'True: {labels[i]}\nPred: {predictions[i]}', color=color, fontsize=10)

    plt.suptitle(f'{model_name} - Sample Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_samples.png', dpi=100)
    plt.close()


def plot_model_comparison(results):
    """Plot and save model comparison"""
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    parameters = [results[m]['parameters'] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    bars1 = axes[0].bar(model_names, accuracies, color=['blue', 'green', 'orange'])
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].set_ylim(95, 100)
    axes[0].grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{acc:.2f}%', ha='center', va='bottom')

    # Parameters comparison
    bars2 = axes[1].bar(model_names, parameters, color=['blue', 'green', 'orange'])
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Model Size Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, param in zip(bars2, parameters):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     f'{param:,}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('MNIST Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=100)
    plt.close()


# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def main():
    """Main function to test all models"""

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print("\nLoading test data...")
    test_loader = load_test_data(batch_size=64)

    # Define models to test
    models_to_test = ['MLP', 'CNN', 'Transformer']

    all_results = {}

    # Test each model
    for model_name in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"TESTING: {model_name}")
        print(f"{'=' * 60}")

        # Load trained model
        model = test_single_model(model_name, device)
        if model is None:
            continue

        # Evaluate model
        accuracy, class_accuracies, predictions, targets = evaluate_model(
            model, test_loader, device, model_name
        )

        # Store results
        all_results[model_name] = {
            'accuracy': accuracy,
            'parameters': model.get_params(),
            'predictions': predictions,
            'targets': targets,
            'class_accuracies': class_accuracies,
            'model': model
        }

        # Generate visualizations
        plot_confusion_matrix(targets, predictions, model_name)
        plot_sample_predictions(model, test_loader, device, model_name)

        # Print classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(targets, predictions, digits=3))

        # Save detailed results
        with open(f'results/{model_name}_results.txt', 'w') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"{model_name} Test Results\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Test Accuracy: {accuracy:.2f}%\n")
            f.write(f"Error Rate: {100 - accuracy:.2f}%\n")
            f.write(f"Correct Predictions: {(predictions == targets).sum()}/{len(targets)}\n")
            f.write(f"Parameters: {model.get_params():,}\n\n")

            f.write("Per-digit Accuracy:\n")
            for digit, acc in enumerate(class_accuracies):
                f.write(f"  Digit {digit}: {acc:.1f}%\n")

            f.write(f"\nClassification Report:\n")
            f.write(classification_report(targets, predictions, digits=3))

        print(f"✓ Results saved to: results/{model_name}_results.txt")

    # Generate comparison plot if we have results
    if all_results:
        plot_model_comparison(all_results)

        # Print final summary
        print(f"\n{'=' * 60}")
        print("TESTING SUMMARY")
        print(f"{'=' * 60}")

        with open('results/testing_summary.txt', 'w') as f:
            f.write("MNIST Testing Summary\n")
            f.write("=" * 50 + "\n\n")

            for model_name, results in all_results.items():
                print(f"{model_name:12s}: {results['accuracy']:.2f}% "
                      f"(Error: {100 - results['accuracy']:.2f}%, "
                      f"Params: {results['parameters']:,})")

                f.write(f"{model_name}:\n")
                f.write(f"  Test Accuracy: {results['accuracy']:.2f}%\n")
                f.write(f"  Error Rate: {100 - results['accuracy']:.2f}%\n")
                f.write(f"  Parameters: {results['parameters']:,}\n\n")

        print(f"\n✓ All results saved to 'results/' folder")
        print(f"✓ Model comparison saved to 'results/model_comparison.png'")
        print(f"✓ Testing summary saved to 'results/testing_summary.txt'")
    else:
        print("\n✗ No models were tested successfully.")
        print("Make sure to run 'python train.py' first!")


if __name__ == '__main__':
    main()