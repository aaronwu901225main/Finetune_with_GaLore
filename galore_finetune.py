import os, json, uuid, torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from galore_torch import GaLoreAdamW8bit
from trl import setup_chat_format
from tqdm import tqdm

# ========= 超參數 =========
lr = 1e-5
rank = 1024
update_proj_gap = 200
scale = 2
num_training_steps = 1000
warmup_steps = 50
batch_size = 1
max_seq_length = 256

modelpath = "Salesforce/Llama-xLAM-2-8b-fc-r"
run_id = f"galore-{str(uuid.uuid4())}"
save_dir = f"out_{run_id}"

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 模型 & tokenizer =========
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    torch_dtype=torch.bfloat16,
    device_map=None,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)
model, tokenizer = setup_chat_format(model, tokenizer)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.to(device)

# ========= 數據處理 =========
def format_example(example):
    parts = []
    for m in example["messages"]:
        role = m["role"]
        if role == "user":
            parts.append(f"<|im_start|>user\n{m['content_text']}<|im_end|>")
        elif role == "assistant":
            if m["content_text"].strip():
                parts.append(f"<|im_start|>assistant\n{m['content_text']}<|im_end|>")
            elif m.get("tool_uses"):
                parts.append(f"<|im_start|>assistant\n<tool_call>{m['tool_uses']}</tool_call><|im_end|>")
        elif role == "tool":
            if m.get("tool_results"):
                parts.append(f"<|im_start|>tool\n{m['tool_results'][0]}<|im_end|>")
    return {"text": "\n".join(parts)}

dataset = load_dataset("json", data_files="merged.jsonl")["train"].map(format_example)

# 確認處理結果
print(dataset[0]["text"][:1000])

def collate_fn(batch):
    texts = [b["text"] for b in batch]
    toks = tokenizer(texts, max_length=max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
    labels = toks["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    toks["labels"] = labels
    return {k: v.to(device) for k, v in toks.items()}

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# ========= Per-layer GaLore Optimizer =========
galore_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"])]
id_galore_params = [id(p) for p in galore_params]

optimizer_dict, scheduler_dict = {}, {}
for p in model.parameters():
    if not p.requires_grad: continue
    if id(p) in id_galore_params:
        optimizer_dict[p] = GaLoreAdamW8bit(
            [{'params':[p], 'rank':rank, 'update_proj_gap':update_proj_gap*2, 'scale':scale, 'proj_type':'std'}],
            lr=lr, weight_decay=0
        )
    else:
        optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=lr, weight_decay=0)

    # 建立 scheduler（這裡簡單線性 warmup→cosine）
    scheduler_dict[p] = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_dict[p], T_max=num_training_steps*2
    )

def optimizer_hook(p):
    if p.grad is None: return
    optimizer_dict[p].step()
    optimizer_dict[p].zero_grad()
    scheduler_dict[p].step()

for p in model.parameters():
    if p.requires_grad:
        p.register_post_accumulate_grad_hook(optimizer_hook)

# ========= 訓練 loop =========
global_step = 0
model.train()
os.makedirs(save_dir, exist_ok=True)

for epoch in range(3):  # 多跑幾個 epoch
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        global_step += 1
        loss = model(**batch).loss
        loss.backward()
        print(f"Step {global_step} | Loss {loss.item():.4f}")

        if global_step % 50 == 0:
            model.save_pretrained(f"{save_dir}/checkpoint-{global_step}")
            tokenizer.save_pretrained(f"{save_dir}/checkpoint-{global_step}")

        if global_step >= num_training_steps:
            break
    if global_step >= num_training_steps:
        break

print("Training finished.")
