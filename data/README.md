# Data

## SUN397

[SUN397](https://pytorch.org/vision/stable/generated/torchvision.datasets.SUN397.html) is a scene understanding dataset with 397 indoor and outdoor categories — covering the full range of environments an iPhone camera encounters.

**Why SUN397?**
SUN397 contains real scenes (beach, forest, mountain, kitchen, street, office...) at sufficient resolution for ViT's 224×224 input. Unlike object datasets, every image is a scene — exactly what a Smart Scene Classifier should be trained on.

**Download (torchvision — official PyTorch, auto-verified via MD5):**
```python
from torchvision.datasets import SUN397
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = SUN397(root='./data', download=True, transform=transform)
```

**10 classes used (camera-relevant, indoor + outdoor):**
```
beach · forest · mountain · kitchen · bedroom · street · restaurant · office · living_room · park
```

Filtered in notebook `03_train_and_compare.ipynb` — the dataloader selects these 10 classes automatically.

**Stats (10-class subset):**
- ~750 training images per class (~7,500 total)
- ~50 validation images per class (~500 total)
- Image size: variable → resized and center cropped to 224×224
