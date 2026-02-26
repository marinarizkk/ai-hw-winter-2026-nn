"""
test.py - Test and attack MNIST models with FGSM, I-FGSM, MI-FGSM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

print("=" * 70)
print("MNIST TESTING & ATTACKS")
print("=" * 70)

# ============================================================================
# MODEL DEFINITIONS - MUST EXACTLY MATCH train.py
# ============================================================================

class MLP(nn.Module):
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

class CNN(nn.Module):
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

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.patch_size = 7
        self.num_patches = (28 // 7) ** 2
        patch_dim = 1 * 7 * 7

        # Improved patch embedding with LayerNorm and GELU
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        # Smaller initialization for positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, 128) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128) * 0.02)

        # Transformer encoder with GELU
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        b = x.shape[0]
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

# ============================================================================
# ATTACK CLASSES
# ============================================================================

class FGSM:
    def __init__(self, model, eps=0.3):
        self.model, self.eps = model, eps

    def __call__(self, x, y):
        x_adv = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(self.model(x_adv), y)
        self.model.zero_grad()
        loss.backward()
        x_adv = x_adv + self.eps * x_adv.grad.sign()
        return torch.clamp(x_adv, -2.5, 2.5).detach()

class IFGSM:
    def __init__(self, model, eps=0.3, alpha=0.05, iters=10):
        self.model, self.eps, self.alpha, self.iters = model, eps, alpha, iters

    def __call__(self, x, y):
        x_adv = x.clone().detach()
        for _ in range(self.iters):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            self.model.zero_grad()
            loss.backward()
            x_adv = x_adv + self.alpha * x_adv.grad.sign()
            perturbation = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + perturbation, -2.5, 2.5).detach()
        return x_adv

class MIFGSM:
    def __init__(self, model, eps=0.3, alpha=0.05, iters=10, decay=1.0):
        self.model, self.eps, self.alpha, self.iters, self.decay = model, eps, alpha, iters, decay

    def __call__(self, x, y):
        x_adv = x.clone().detach()
        momentum = torch.zeros_like(x)
        for _ in range(self.iters):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad / (x_adv.grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-8)
            momentum = self.decay * momentum + grad
            x_adv = x_adv + self.alpha * momentum.sign()
            perturbation = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + perturbation, -2.5, 2.5).detach()
        return x_adv

# ============================================================================
# DATA LOADING
# ============================================================================

def get_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size, shuffle=False
    )

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, loader, device, attack=None, attack_name='Clean'):
    model.eval()
    correct = total = 0

    for x, y in tqdm(loader, desc=f'{attack_name:10s}'):
        x, y = x.to(device), y.to(device)
        if attack:
            x = attack(x, y)
        with torch.no_grad():
            _, pred = model(x).max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return 100 * correct / total

def test_epsilon_range(model, loader, device, epsilons):
    attacks = {
        'FGSM': FGSM,
        'I-FGSM': lambda m, e: IFGSM(m, e, e/5, 10),
        'MI-FGSM': lambda m, e: MIFGSM(m, e, e/5, 10)
    }

    results = {name: [] for name in attacks}

    for eps in epsilons:
        print(f"\nEpsilon = {eps:.2f}")
        for name, attack_class in attacks.items():
            attack = attack_class(model, eps)
            acc = evaluate(model, loader, device, attack, name)
            results[name].append(100 - acc)  # ASR
            print(f"  {name:8s}: ASR {100-acc:.2f}%")

    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def show_examples(model, loader, device, model_name):
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:8].to(device), y[:8].to(device)

    attacks = [
        ('Clean', None),
        ('FGSM', FGSM(model, 0.3)),
        ('I-FGSM', IFGSM(model, 0.3, 0.06, 10)),
        ('MI-FGSM', MIFGSM(model, 0.3, 0.06, 10))
    ]

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))

    for row, (name, attack) in enumerate(attacks):
        if attack is None:
            x_adv = x
        else:
            x_adv = attack(x, y)

        with torch.no_grad():
            pred = model(x_adv).argmax(1)

        for col in range(8):
            img = x_adv[col].cpu().numpy().squeeze()
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            color = 'green' if pred[col] == y[col] else 'red'
            axes[row, col].set_title(f'{pred[col].item()}', color=color, fontsize=10)

        axes[row, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=40, ha='right')

    plt.suptitle(f'{model_name} - Adversarial Examples (ε=0.3)', fontsize=14)
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_examples.png')
    plt.close()

def plot_results(model_name, epsilons, results):
    plt.figure(figsize=(10,5))
    for name, asr in results.items():
        plt.plot(epsilons, asr, 'o-', linewidth=2, markersize=8, label=name)

    plt.xlabel('Epsilon')
    plt.ylabel('Attack Success Rate (%)')
    plt.title(f'{model_name} - ASR vs Epsilon')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 105)

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_asr.png')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    loader = get_test_loader()
    print(f"Test samples: {len(loader.dataset)}")

    # Check what model files actually exist
    import glob
    model_files = glob.glob("*_model.pth")
    print(f"\nFound model files: {model_files}")

    models = [
        ('MLP', MLP(), 'MLP_model.pth'),
        ('CNN', CNN(), 'CNN_model.pth'),
        ('Transformer', Transformer(), 'Transformer_model.pth')
    ]

    epsilons = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for name, model_class, model_file in models:
        print(f"\n{'='*60}\n{name}\n{'='*60}")

        if not os.path.exists(model_file):
            print(f"✗ {model_file} not found. Skipping {name}...")
            continue

        try:
            state_dict = torch.load(model_file, map_location=device)
            model_class.load_state_dict(state_dict)
            model = model_class.to(device)
            print(f"✓ Successfully loaded {model_file}")
        except Exception as e:
            print(f"✗ Error loading {model_file}: {e}")
            continue

        # Clean accuracy
        clean_acc = evaluate(model, loader, device, attack=None, attack_name='Clean')
        print(f"\nClean Accuracy: {clean_acc:.2f}%")

        # Test epsilon range
        results = test_epsilon_range(model, loader, device, epsilons)

        # Save results
        os.makedirs('results', exist_ok=True)
        with open(f'results/{name}_attack_results.txt', 'w') as f:
            f.write(f"{'='*60}\n{name} Attack Results\n{'='*60}\n")
            f.write(f"Clean Accuracy: {clean_acc:.2f}%\n\n")
            f.write("ASR (%):\nEpsilon ")
            for attack in results: f.write(f"{attack:>10}")
            f.write("\n" + "-"*50 + "\n")
            for i, eps in enumerate(epsilons):
                f.write(f"{eps:.2f}   ")
                for attack in results:
                    f.write(f"{results[attack][i]:10.2f}")
                f.write("\n")

        # Plot
        plot_results(name, epsilons, results)
        show_examples(model, loader, device, name)
        print(f"✓ Results saved for {name}")

    print(f"\n{'='*60}")
    print("All done! Results saved to results/ folder")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()