# Data

## ImageNette

ImageNette is a 10-class subset of ImageNet with 224×224 images — the native input size for ViT.

**Download:**
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
```

Place the extracted folder here: `data/imagenette2-320/`

**Structure after extraction:**
```
data/
└── imagenette2-320/
    ├── train/
    │   ├── n01440764/   (tench)
    │   ├── n02102040/   (English springer)
    │   └── ...
    └── val/
        └── ...
```

**Classes:** tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute
