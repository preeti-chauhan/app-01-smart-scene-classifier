# Data

## SUN397

[SUN397](https://vision.princeton.edu/projects/2010/SUN/) is a scene understanding dataset with 397 indoor and outdoor categories — covering the full range of environments an iPhone camera encounters.

**Why SUN397?**
SUN397 contains real scenes (beach, forest, mountain, kitchen, street, office...) at sufficient resolution for ViT's 224×224 input. Unlike object datasets, every image is a scene — exactly what a Smart Scene Classifier should be trained on.

**Download (HuggingFace — reliable CDN, no license prompt):**
```python
from datasets import load_dataset
ds = load_dataset("sun397", split="train")
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
