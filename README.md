# Smart Scene Classifier

Real-time scene classification on iPhone using Vision Transformer — with Grad-CAM visualization showing which regions of the image drove the prediction.

---

## Results

### Training — Scratch vs Pretrained ViT
Fine-tuning a pretrained ViT-B/16 converges faster and reaches higher accuracy than training from scratch on the same 10-class SUN397 subset — demonstrating the value of ImageNet pretraining for small datasets.

![Training curves](assets/training_curves.png)

### CoreML Inference — 10-class predictions
Each image is classified by the exported CoreML model running on Mac. Labels in green are correct predictions; red are misclassifications. Confidence shown as the softmax probability of the top class.

![Predictions](assets/coreml_predictions.png)

### Latency Benchmark
Single-image inference time across CoreML compute unit configurations vs PyTorch on MPS. ALL routes ops across Neural Engine, GPU, and CPU automatically. Results measured on Apple Silicon (50 runs after 10 warmup).

![Benchmark](assets/coreml_benchmark.png)

## Notebooks

> **Note:** If running in Colab, local filesystem resets when the session ends. Run cells in order — Drive is handled automatically where needed.

| Notebook | Description |
|---|---|
| `01_self_attention.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/01_self_attention.ipynb) | Scaled dot-product attention and multi-head attention from scratch |
| `02_vit_from_scratch.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/02_vit_from_scratch.ipynb) | Full ViT: patch embeddings, CLS token, positional encoding, transformer encoder |
| `03_train_and_compare.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/03_train_and_compare.ipynb) | Train ViT from scratch vs fine-tune pretrained ViT on SUN397 — saves checkpoints to Google Drive |
| `04_gradcam_visualization.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/04_gradcam_visualization.ipynb) | Grad-CAM and attention maps — loads checkpoint from Google Drive |
| `05_coreml_export.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/app-01-smart-scene-classifier/blob/main/notebooks/05_coreml_export.ipynb) | Export to CoreML, benchmark on Apple Neural Engine — loads checkpoint from Google Drive |

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
