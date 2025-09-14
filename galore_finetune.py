# %% [markdown]
# # GaLore

# %% [markdown]
# GaLore is a novel approach detailed in the publication GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. 
# GaLore reduces VRAM requirements not by decreasing the number of parameters directly but by optimizing how these parameters are trained.
# 
# GaLore focuses on two primary strategies:
# * Gradient Low-Rank Projection: GaLore shifts away from handling the full, high-dimensional gradients of weight matrices. Instead, it projects these gradients onto a low-rank space, significantly reducing the computational load while retaining essential informat
# * Per-Layer Weight Updates: Unlike the conventional method where an optimizer updates all layers simultaneously after backpropagation, GaLore implements updates on a per-layer basis during backpropagation. This approach further reduces the memory footprint throughout the training process.
# 
# Paper: https://huggingface.co/papers/2403.03507
# 
# Official HF blog post: https://huggingface.co/blog/galore
# 
# My experiments: https://medium.com/@geronimo7/llm-training-on-consumer-gpus-with-galore-d25075143cfb

# %% [markdown]
# **Here's how to train the llama2-7b model using the GaLore layerwise method, operational on a single NVIDIA GeForce RTX 3090 GPU.**

# %% [markdown]
# # Prerequisistes

# %%
!pip install -U bitsandbytes wandb huggingface_hub flash_attn sentencepiece accelerate datasets trl transformers tokenizers galore_torch flash_attn

# %% [markdown]
# Versions known to work:
# ```
# accelerate==0.28.0
# bitsandbytes==0.43.0
# datasets==2.18.0
# tokenizers==0.15.2
# transformers==4.39.1
# trl==0.8.1
# wandb==0.16.4
# torch==2.2.1
# ```

# %% [markdown]
# # Log into wandb and huggingface

# %%
from wandb import login
login(key="XXX")

# %%
import huggingface_hub
huggingface_hub.login(token="XXX")

# %% [markdown]
# # Setup GaLore hyperparameters

# %% [markdown]
# The GaLore optimizer comes with a few hyperparameters to set:
# 
# * `target_modules_list`: Specifies the layers targeted by GaLore
# * `rank`: The rank of the projection matrices. Similar to LoRA, the higher the rank the more closely the finetuning will resemble a full parameter finetune. The GaLore authors recomment 1024 for a 7B model.
# * `update_proj_gap`: The number of steps after which the projections are updated. The update is an expensive step and takes around 15 minutes for a 7B model. Defines the interval for updating projections, with a suggested range between 50 and 1000 steps.
# * `scale`: A scale factor akin to LoRA’s alpha, adjusting the update strength. After trying a few values I found scale=2 to most closely resemble a classic full-parameter finetune.

# %%
lr = 1e-5

rank = 1024
update_proj_gap = 200
scale = 2

# %% [markdown]
# # Load model, tokenizer, dataset and setup trainer

# %%
# change me!
modelpath = "meta-llama/Llama-2-7b-hf"

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed, get_constant_schedule
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, uuid, wandb

set_seed(42)
run_id = f"galore-{str(uuid.uuid4())}"

model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    torch_dtype = torch.bfloat16,
    attn_implementation = "flash_attention_2",  
    device_map = "auto",
    use_cache = False,
)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

model, tokenizer = setup_chat_format(model, tokenizer)
if tokenizer.pad_token in [None, tokenizer.eos_token]: 
    tokenizer.pad_token = tokenizer.unk_token

dataset = load_dataset("g-ronimo/oasst2_top4k_en")

# %%
training_arguments = TrainingArguments(
    output_dir = f"out_{run_id}",
    evaluation_strategy = "steps",
    label_names = ["labels"],
    per_device_train_batch_size = 16,
    save_steps = 250,
    eval_steps = 250,
    logging_steps = 1, 
    learning_rate = lr,
    num_train_epochs = 3,
    lr_scheduler_type = "constant",
    gradient_checkpointing = True,
    group_by_length = False,
    optim="galore_adamw_8bit_layerwise",
    optim_target_modules=["attn", "mlp"],
    optim_args=f"rank={rank}, update_proj_gap={update_proj_gap}, scale={scale}",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset['test'],
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template = "<|im_start|>user", 
        response_template = "<|im_start|>assistant", 
        tokenizer = tokenizer, 
        mlm = False),
    max_seq_length = 256,
    dataset_kwargs = dict(add_special_tokens = False),
    args = training_arguments,
)

# %% [markdown]
# # Init wandb and start training

# %%
wandb.init(
    project = "galore-7B", 
    name = run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()


