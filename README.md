# Latent consistency distillation

This script is used to train a model with latent consistency distillation. Fixed some bugs from https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl_wds.py


## ⭐ library version
```
cuda: 11.7
pytorch: 2.0.1+cu117
transformers: 4.35.0
diffusers: 0.23.0
accelerate: 0.24.1
peft: 0.6.1
webdataset: 0.2.69
```

## Bug1: Training stuck at 0%
👉https://github.com/huggingface/diffusers/issues/5743

```bash
11/17/2023 02:35:08 - INFO - __main__ - ***** Running training *****
11/17/2023 02:35:08 - INFO - __main__ -   Num batches each epoch = 1752
11/17/2023 02:35:08 - INFO - __main__ -   Num Epochs = 1
11/17/2023 02:35:08 - INFO - __main__ -   Instantaneous batch size per device = 1
11/17/2023 02:35:08 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 4
11/17/2023 02:35:08 - INFO - __main__ -   Gradient Accumulation steps = 4
11/17/2023 02:35:08 - INFO - __main__ -   Total optimization steps = 1
Steps:   0%|                                                                                                                                                       | 0/1 [00:00<?, ?it/s]
```

#### ⭐ Modify the class Text2ImageDataset
    
```python
class Text2ImageDataset:
    ...
    ...
    def get_orig_size(json, resolution=1024, use_fix_crop_and_size=False):
        if use_fix_crop_and_size:
            return (resolution, resolution, str(json.get("caption", "")))
        else:
            return (int(json.get("original_width", 0.0)), int(json.get("original_height", 0.0)),
                    str(json.get("caption", "")))

    processing_pipeline = [
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", orig_size="json", handler=wds.warn_and_continue),
        wds.map(filter_keys({"image", "orig_size"})),
        wds.map_dict(orig_size=custom_get_orig_size),
        wds.map(transform),
        wds.to_tuple("image", "orig_size", "crop_coords"),
    ]

    # Create train dataset and loader
    pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            # wds.select(WebdatasetFilter(min_size=960)),
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]
    ...
    ...

def main(args):
    ...
    ...
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                image, text, orig_size, crop_coords = batch[0], batch[1][2], [batch[1][0], batch[1][1]], batch[2]
    ...
    ...
```

#### 📕 Dataset structure
```bash
├── train_dataset
│   ├── 000000.tar
│   │   ├── 000000.jpg
│   │   ├── 000000.json
│   │   ├── 000001.jpg
│   │   ├── 000001.json
│   │   ├── ...
│   ├── 000001.tar
│   │   ├── 000000.jpg
│   │   ├── 000000.json
│   │   ├── ...
```
**json file structure**
```json
{
    "original_width": 900,
    "original_height": 1200,
    "pwatermark": 0.0,
    "caption": "SongYun, hanfu, Song style, summer, dizzy, gradual change, wide sleeves, beizi, one-piece, pleated skirt, suspenders,"
}
```
## Bug1: Can't load Lora model

**Before repair**
```
unet.base_model.model.down_blocks.0.downsamplers.0.conv.lora_A.weight
unet.base_model.model.down_blocks.0.downsamplers.0.conv.lora_B.weight
unet.base_model.model.down_blocks.0.resnets.0.conv1.lora_A.weight
unet.base_model.model.down_blocks.0.resnets.0.conv1.lora_B.weight
unet.base_model.model.down_blocks.0.resnets.0.conv2.lora_A.weight
unet.base_model.model.down_blocks.0.resnets.0.conv2.lora_B.weight
unet.base_model.model.down_blocks.0.resnets.0.time_emb_proj.lora_A.weight
unet.base_model.model.down_blocks.0.resnets.0.time_emb_proj.lora_B.weight
...
Total of 846 lora layers
```
**When load Lora model, it will raise an error:**
```bash
AttributeError: 'UNet2DConditionModel' object has no attribute 'base_model'
```

#### ⭐ Modify the save method

```python
# Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
        output_path = os.path.join(args.output_dir, "unet_lora")
        StableDiffusionXLPipeline.save_lora_weights(output_path, lora_state_dict)

        # delete the "unet." from model state dict keys
        tensors = {}
        model_path = os.path.join(output_path, "pytorch_lora_weights.safetensors")
        with safe_open(model_path, framework="pt", device=0) as f:
            for k in f.keys():
                new_key = k.replace("unet.", "", 1)
                tensors[new_key] = f.get_tensor(k)
        save_file(tensors, model_path)
```
**After repair**
```
lora_unet_down_blocks_0_downsamplers_0_conv.alpha
lora_unet_down_blocks_0_downsamplers_0_conv.lora_down.weight
lora_unet_down_blocks_0_downsamplers_0_conv.lora_up.weight
lora_unet_down_blocks_0_resnets_0_conv1.alpha
lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight
lora_unet_down_blocks_0_resnets_0_conv1.lora_up.weight
lora_unet_down_blocks_0_resnets_0_conv2.alpha
...
Total of 1269 lora layers
```