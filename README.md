# Smart Scene Classifier

Real-time scene classification on iPhone using Vision Transformer — with Grad-CAM visualization showing which regions of the image drove the prediction.

---

## Demo

*Screenshots and screen recording coming after training.*

## Notebooks

| Notebook | Description | Run |
|---|---|---|
| `01_self_attention.ipynb` | Scaled dot-product attention and multi-head attention from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/01_self_attention.ipynb) |
| `02_vit_from_scratch.ipynb` | Full ViT: patch embeddings, CLS token, positional encoding, transformer encoder | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/02_vit_from_scratch.ipynb) |
| `03_train_and_compare.ipynb` | Train ViT from scratch vs fine-tune pretrained ViT on SUN397 — accuracy comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/03_train_and_compare.ipynb) |
| `04_gradcam_visualization.ipynb` | Grad-CAM and attention maps — visualizing what the model sees | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/04_gradcam_visualization.ipynb) |
| `05_coreml_export.ipynb` | Export to CoreML, benchmark on Apple Neural Engine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/05_coreml_export.ipynb) |

## Running in Colab

Colab's local filesystem resets when the session ends. Each notebook includes a Drive mount cell and saves/loads checkpoints from `/content/drive/MyDrive/app-01/` automatically — just run the cells in order.

## Stack

| Tool | Role |
|---|---|
| [PyTorch](https://pytorch.org) | Model training (MPS backend — Apple Silicon) |
| [timm](https://huggingface.co/docs/timm) | Pretrained ViT weights for fine-tuning |
| [einops](https://einops.rocks) | Readable tensor operations |
| [coremltools](https://apple.github.io/coremltools) | Export to CoreML for on-device inference |
| [matplotlib](https://matplotlib.org) | Grad-CAM and attention map visualization |

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 images across 10 classes (50k train / 10k test): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Images are resized to 224×224 for ViT. Downloads automatically via `torchvision.datasets.CIFAR10` (~170 MB).
