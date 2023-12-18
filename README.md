# Latent consistency distillation

**This script is used to train a model with latent consistency distillation.**

- **test_train_lcm_lora_bucket.py** is not memory optimization
- **train_lcm_distill_sdxl_lora.py.py** with memory optimization

#### Reference
https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl_wds.py

https://github.com/huggingface/diffusers/pull/5778

https://github.com/NovelAI/novelai-aspect-ratio-bucketing

#### Midfy&Add:
- **Fix the dataset loading issue.**
- **Fix the LoRA weight saving issue.**
- **Reduce memory consumption during training.**
- **Add an aspect-ratio-bucketing option.**

#### 📕 Dataset structure
```bash
├── train_datasetr
│   ├── 000000.jpg
│   ├── 000000.caption
│   ├── 000001.jpg
│   ├── 000001.caption
│   ├── 000002.jpg
│   ├── 000002.caption
│   │...
```

