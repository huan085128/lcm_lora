import subprocess

# ----------Model Checkpoint Loading Arguments----------
pretrained_teacher_model = "stabilityai/stable-diffusion-xl-base-1.0"
pretrained_vae_model_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
train_data_dir = "path/to/datasets"
output_dir = "path/to/saved/model"

# ----------Training Arguments----------
resolution = 1024
center_crop = False
random_flip = False
xformers = True
gradient_checkpointing = False
use_aspect_ratio_bucket = True
use_8bit_adam = True

max_train_steps = 10000

mixed_precision = "fp16"
dataloader_num_workers = 0
train_batch_size = 2
gradient_accumulation_steps = 1
num_train_epochs = 30
checkpointing_steps = 5000
learning_rate = 1e-06
lr_scheduler = "constant"
lr_warmup_steps = 0
report_to = "wandb"
validation_steps = 1

seed = 42

# ----bucketing parameters----
debug_prompt = False
debug_arb = False
arb_dim_limit = 1024
arb_min_dim = 512
arb_divisible = 64
arb_max_ar_error = 4

# ----Latent Consistency Distillation (LCD) Specific Arguments----
w_min = 3.0
w_max = 15.0
num_ddim_timesteps = 50
loss_type = "l2"
huber_c = 0.001
lora_rank = 128

# Construct arguments list for the training script
train_args = [
    f"--pretrained_teacher_model={pretrained_teacher_model}",
    f"--train_data_dir={train_data_dir}",
    f"--output_dir={output_dir}",
    f"--resolution={resolution}",
    f"--dataloader_num_workers={dataloader_num_workers}",
    "--center_crop" if center_crop else "",
    "--random_flip" if random_flip else "",
    "--enable_xformers_memory_efficient_attention" if xformers else "",
    f"--train_batch_size={train_batch_size}",
    f"--gradient_accumulation_steps={gradient_accumulation_steps}",
    "--use_aspect_ratio_bucket" if use_aspect_ratio_bucket else "",
    "--gradient_checkpointing" if gradient_checkpointing else "",
    "--use_8bit_adam" if use_8bit_adam else "",
    f"--max_train_steps={max_train_steps}",
    f"--num_train_epochs={num_train_epochs}",
    f"--checkpointing_steps={checkpointing_steps}",
    f"--learning_rate={learning_rate}",
    f"--lr_scheduler={lr_scheduler}",
    f"--lr_warmup_steps={lr_warmup_steps}",
    f"--report_to={report_to}",
    f"--mixed_precision={mixed_precision}",
    f"--validation_steps={validation_steps}",
    f"--seed={seed}",
    "--debug_prompt" if debug_prompt else "",
    "--debug_arb" if debug_arb else "",
    f"--arb_dim_limit={arb_dim_limit}",
    f"--arb_min_dim={arb_min_dim}",
    f"--arb_divisible={arb_divisible}",
    f"--arb_max_ar_error={arb_max_ar_error}",
    f"--w_min={w_min}",
    f"--w_max={w_max}",
    f"--num_ddim_timesteps={num_ddim_timesteps}",
    f"--loss_type={loss_type}",
    f"--huber_c={huber_c}",
    f"--lora_rank={lora_rank}",
]

train_script = "test_train_lcm_lora_bucket.py"

train_args = [arg for arg in train_args if arg]

cmd = (
    f"accelerate launch --mixed_precision={mixed_precision} {train_script} " +
    " ".join(train_args)
)

print(f"Running command: {cmd}")
subprocess.run(cmd, shell=True, check=True)
