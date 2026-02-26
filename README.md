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

[ImageNette](https://github.com/fastai/imagenette) — 10-class [ImageNet](https://www.image-net.org) subset, 224×224. See `data/README.md` for download instructions.
