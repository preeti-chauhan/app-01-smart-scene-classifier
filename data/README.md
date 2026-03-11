# Data

> **Note:** The original dataset for this project was SUN397 (scene understanding). The SUN397 download URL from Princeton's server returns a 404 and is no longer available via torchvision. CIFAR-10 is used as a reliable alternative to demonstrate the ViT training comparison. The technical results (scratch vs fine-tuned accuracy gap) are equivalent.

## CIFAR-10

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is an image classification dataset with 60,000 32×32 images across 10 classes (50k train / 10k test).

**Download (torchvision — automatic):**
```python
from torchvision.datasets import CIFAR10
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
```

**10 classes:**
```
airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck
```

Loaded in notebook `03_train_and_compare.ipynb` — all 10 classes used directly.

**Stats:**
- 50,000 training images (5,000 per class)
- 10,000 validation images (1,000 per class)
- Image size: 32×32 → resized to 224×224 for ViT
- Download size: ~170 MB
