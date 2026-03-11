# Data

## SUN397

[SUN397](https://3dvision.princeton.edu/projects/2010/SUN/) is a scene understanding dataset with 397 indoor and outdoor categories — covering the full range of environments an iPhone camera encounters.

Loaded via HuggingFace datasets from `pc-ml-dl/sun397`. No manual download required.

**Load in Python:**
```python
from datasets import load_dataset
train = load_dataset('pc-ml-dl/sun397', split='train')
test  = load_dataset('pc-ml-dl/sun397', split='test')
```

**10 classes used (camera-relevant, indoor + outdoor):**
```
beach · forest · mountain · kitchen · bedroom · street · restaurant · office · living room · park
```

Filtered in notebook `03_train_and_compare.ipynb` using `SUN397Subset`.

**Stats (10-class subset):**
- ~1,000 training images per class (~10,000 total)
- ~500 test images per class (~5,000 total)
- Image size: variable → resized and center cropped to 224×224

> **Note:** The original torchvision `SUN397(download=True)` is broken — the Princeton download URL returns a 403. The dataset is hosted on HuggingFace at `pc-ml-dl/sun397` as a workaround.
