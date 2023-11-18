import subprocess

# Set parameters
pretrained_teacher_model = "/mnt/private/models/stable-diffusion-xl-base-1.0"
train_shards_path_or_url = "/mnt/private/lcm_finetune/train_test3.tar"
output_dir = "/mnt/private/lcm_finetune/lcm-xl-distilled1"

resolution = 1024
center_crop = False
random_flip = False
xformers = True
use_fix_crop_and_size = False
gradient_checkpointing = False
use_8bit_adam = False

max_train_steps = 10000

mixed_precision = "fp16"
dataloader_num_workers = 8
max_train_samples = 1746
train_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs = 30
checkpointing_steps = 1000
learning_rate = 1e-04
lr_scheduler = "constant"
lr_warmup_steps = 0
report_to = "tensorboard"
validation_steps = 10

# ----Latent Consistency Distillation (LCD) Specific Arguments----
w_min = 3.0
w_max = 15.0
num_ddim_timesteps = 50
loss_type = "l2"
huber_c = 0.001
lora_rank = 64

# Construct arguments list for the training script
train_args = [
    f"--pretrained_teacher_model={pretrained_teacher_model}",
    f"--train_shards_path_or_url={train_shards_path_or_url}",
    f"--output_dir={output_dir}",
    f"--resolution={resolution}",
    f"--dataloader_num_workers={dataloader_num_workers}",
    f"--max_train_samples={max_train_samples}",
    "--center_crop" if center_crop else "",
    "--random_flip" if random_flip else "",
    "--enable_xformers_memory_efficient_attention" if xformers else "",
    "--use_fix_crop_and_size" if use_fix_crop_and_size else "",
    f"--train_batch_size={train_batch_size}",
    f"--gradient_accumulation_steps={gradient_accumulation_steps}",
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
    f"--w_min={w_min}",
    f"--w_max={w_max}",
    f"--num_ddim_timesteps={num_ddim_timesteps}",
    f"--loss_type={loss_type}",
    f"--huber_c={huber_c}",
    f"--lora_rank={lora_rank}",
]

train_script = "train_lcm_distill_lora_sdxl_wds.py"

train_args = [arg for arg in train_args if arg]

cmd = (
    f"accelerate launch --mixed_precision={mixed_precision} {train_script} " +
    " ".join(train_args)
)

print(f"Running command: {cmd}")
subprocess.run(cmd, shell=True, check=True)
