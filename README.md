# Smart Scene Classifier

Real-time scene classification on iPhone using Vision Transformer — with Grad-CAM visualization showing which regions of the image drove the prediction.

[![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/preeti-chauhan/app-01-smart-scene-classifier&count_bg=%2379C83D&title_bg=%23555555&title=views&edge_flat=false)](https://github.com/preeti-chauhan/app-01-smart-scene-classifier)

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

## Stack

| Tool | Role |
|---|---|
| [PyTorch](https://pytorch.org) | Model training (MPS backend — Apple Silicon) |
| [timm](https://huggingface.co/docs/timm) | Pretrained ViT weights for fine-tuning |
| [einops](https://einops.rocks) | Readable tensor operations |
| [coremltools](https://apple.github.io/coremltools) | Export to CoreML for on-device inference |
| [matplotlib](https://matplotlib.org) | Grad-CAM and attention map visualization |

## Dataset

[SUN397](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.SUN397.html) — 397-category scene understanding dataset covering indoor and outdoor environments. A 10-class camera-relevant subset is used for training: beach, forest, mountain, kitchen, bedroom, street, restaurant, office, living room, park. Downloaded via `torchvision.datasets.SUN397`. See `data/README.md` for details.
