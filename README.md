# Smart Scene Classifier

Real-time scene classification on iPhone using Vision Transformer — with Grad-CAM visualization showing which regions of the image drove the prediction.

---

## Demo

*Screenshots and screen recording coming after training.*

## Notebooks

Colab's local filesystem resets when the session ends. Run cells in order — Drive is handled automatically where needed.

| Notebook | Description | Drive |
|---|---|---|
| `01_self_attention.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/01_self_attention.ipynb) | Scaled dot-product attention and multi-head attention from scratch | — |
| `02_vit_from_scratch.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/02_vit_from_scratch.ipynb) | Full ViT: patch embeddings, CLS token, positional encoding, transformer encoder | — |
| `03_train_and_compare.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/03_train_and_compare.ipynb) | Train ViT from scratch vs fine-tune pretrained ViT on SUN397 — accuracy comparison | saves checkpoints |
| `04_gradcam_visualization.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/04_gradcam_visualization.ipynb) | Grad-CAM and attention maps — visualizing what the model sees | loads checkpoint |
| `05_coreml_export.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/05_coreml_export.ipynb) | Export to CoreML, benchmark on Apple Neural Engine | loads checkpoint |

## Stack

| Tool | Role |
|---|---|
| [PyTorch](https://pytorch.org) | Model training (MPS backend — Apple Silicon) |
| [timm](https://huggingface.co/docs/timm) | Pretrained ViT weights for fine-tuning |
| [einops](https://einops.rocks) | Readable tensor operations |
| [coremltools](https://apple.github.io/coremltools) | Export to CoreML for on-device inference |
| [matplotlib](https://matplotlib.org) | Grad-CAM and attention map visualization |

## Dataset

[SUN397](https://3dvision.princeton.edu/projects/2010/SUN/) — scene understanding dataset with 397 indoor and outdoor categories. A 10-class camera-relevant subset is used: beach, forest, mountain, kitchen, bedroom, street, restaurant, office, living room, park. Loaded via HuggingFace datasets (`pc-ml-dl/sun397`).
