# Smart Scene Classifier

Real-time scene and object classification on iPhone using Vision Transformer — with Grad-CAM visualization showing which regions of the image drove the prediction.

## Demo

*Screenshots and screen recording coming after training.*

## Notebooks

| Notebook | Description |
|---|---|
| `01_self_attention.ipynb` | Scaled dot-product attention and multi-head attention from scratch |
| `02_vit_from_scratch.ipynb` | Full ViT: patch embeddings, CLS token, positional encoding, transformer encoder |
| `03_train_and_compare.ipynb` | Train ViT from scratch vs fine-tune DINOv2 on ImageNette — accuracy comparison |
| `04_gradcam_visualization.ipynb` | Grad-CAM and attention maps — visualizing what the model sees |
| `05_coreml_export.ipynb` | Export to CoreML, benchmark on Apple Neural Engine |

## Stack

| Tool | Role |
|---|---|
| [PyTorch](https://pytorch.org) | Model training (MPS backend — Apple Silicon) |
| [timm](https://huggingface.co/docs/timm) | DINOv2 pretrained weights |
| [einops](https://einops.rocks) | Readable tensor operations |
| [coremltools](https://apple.github.io/coremltools) | Export to CoreML for on-device inference |
| [matplotlib](https://matplotlib.org) | Grad-CAM and attention map visualization |

## Dataset

[SUN397](https://huggingface.co/datasets/sun397) — 397-category scene understanding dataset covering indoor and outdoor environments. A 10-class camera-relevant subset is used for training: beach, forest, mountain, kitchen, bedroom, street, restaurant, office, living room, park. Downloaded via HuggingFace `datasets`. See `data/README.md` for details.
