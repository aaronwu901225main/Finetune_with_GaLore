#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

# GaLore optimizers (pip install galore-torch)
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit

# -----------------------------
# 參數分組：決定哪些參數用 GaLore
# -----------------------------
EXCLUDE_IN_GALORE = (
    "norm", "ln_", "layer_norm", "layernorm", "rmsnorm", "embeddings", "embed_tokens", "wpe", "lm_head"
)
DEFAULT_GALORE_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
    "dense", "fc", "proj", "mlp"
)

def split_galore_params(
    model: torch.nn.Module,
    galore_keywords: Tuple[str, ...] = DEFAULT_GALORE_TARGETS,
    exclude_keywords: Tuple[str, ...] = EXCLUDE_IN_GALORE,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """將模型參數拆成需要套 GaLore 與不需要的兩組"""
    galore_params, non_galore_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if any(ex in lname for ex in exclude_keywords) or p.ndim < 2:
            non_galore_params.append(p)
        elif any(kw in lname for kw in galore_keywords):
            galore_params.append(p)
        else:
            # 預設：線性類參數也可納入 GaLore（若不要，改丟到 non_galore）
            galore_params.append(p)
    return galore_params, non_galore_params

# -----------------------------
# 資料處理
# -----------------------------
def build_datasets(dataset_path: str, tokenizer, block_size: int) -> DatasetDict:
    """
    支援：
    - 資料夾：包含 train.* / validation.*（text 或 json/jsonl）
    - 檔案：單一 train 檔；若同路徑有 validation.* 會自動載入
    """
    path = Path(dataset_path)
    data_kwargs = {}
    builder = None

    def infer_builder(p: Path):
        if p.suffix.lower() in [".json", ".jsonl"]:
            return "json"
        return "text"

    if path.is_dir():
        # 嘗試抓 train / validation
        files = list(path.iterdir())
        train_file = None
        valid_file = None
        for f in files:
            n = f.name.lower()
            if n.startswith("train."):
                train_file = f
            if n.startswith("validation.") or n.startswith("valid."):
                valid_file = f
        if train_file is None:
            raise FileNotFoundError("資料夾需包含 train.* 檔案（text/json/jsonl）")
        builder = infer_builder(train_file)
        if builder == "json":
            data_kwargs["field"] = None  # 預設 'text' 欄位；若不是，請自行修改下方 map
        data_files = {"train": str(train_file)}
        if valid_file:
            data_files["validation"] = str(valid_file)
        ds = load_dataset(builder, data_files=data_files)
    else:
        builder = infer_builder(path)
        if builder == "json":
            data_kwargs["field"] = None
        ds = load_dataset(builder, data_files={"train": str(path)})

        # 嘗試在同資料夾找 validation
        val_cand = None
        for cand in ["validation.jsonl", "validation.json", "validation.txt", "valid.jsonl", "valid.json", "valid.txt"]:
            if (path.parent / cand).exists():
                val_cand = path.parent / cand
                break
        if val_cand:
            vbuilder = infer_builder(val_cand)
            v = load_dataset(vbuilder, data_files={"validation": str(val_cand)})
            if "validation" in v:
                ds = DatasetDict(train=ds["train"], validation=v["validation"])

    # 將資料整理成 text 欄位
    def to_text(example):
        if "text" in example and isinstance(example["text"], str):
            return {"text": example["text"]}
        # 若是 json 且欄位不是 text，盡量拼出字串
        if builder == "json":
            # 把所有非 None 的值串起來（你也可以改成抓特定欄位）
            parts = []
            for k, v in example.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float)):
                    parts.append(str(v))
            return {"text": " ".join(parts)}
        # text builder 會有 "text"
        return {"text": example.get("text", "")}

    ds = ds.map(to_text, remove_columns=[c for c in ds["train"].column_names if c != "text"])

    # tokenize & group
    def tokenize(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # 將 token 串接後切成固定長度 block
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        input_ids = [concatenated[i:i + block_size] for i in range(0, total_length, block_size)]
        # causal LM labels = input_ids 本體
        return {
            "input_ids": input_ids,
            "labels": input_ids.copy(),
            "attention_mask": [[1] * block_size for _ in range(total_length // block_size)],
        }

    lm_datasets = tokenized.map(group_texts, batched=True)
    return lm_datasets

# -----------------------------
# 優化器建立（一般 / 8bit / 8bit per-layer）
# -----------------------------
def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    galore_rank: int,
    galore_scale: float,
    update_proj_gap: int,
    proj_type: str,
    optimizer_type: str,
) -> Optimizer:
    galore_params, non_galore_params = split_galore_params(model)
    param_groups = [
        {"params": non_galore_params, "weight_decay": weight_decay},
        {
            "params": galore_params,
            "rank": galore_rank,
            "update_proj_gap": update_proj_gap,
            "scale": galore_scale,
            "proj_type": proj_type,
            "weight_decay": weight_decay,
        },
    ]

    if optimizer_type == "galore_adamw":
        return GaLoreAdamW(param_groups, lr=lr)
    elif optimizer_type == "galore_adamw8bit":
        return GaLoreAdamW8bit(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_type for build_optimizer: {optimizer_type}")

def build_per_layer_optimizers(
    model: torch.nn.Module,
    lr: float,
    galore_rank: int,
    galore_scale: float,
    update_proj_gap: int,
    proj_type: str,
    eightbit: bool = True,
) -> Dict[torch.nn.Parameter, Optimizer]:
    """依 README：對每個參數建立一個優化器並註冊 hook。
       注意：此模式僅支援單卡，不建議搭配 AMP。"""
    opt_dict: Dict[torch.nn.Parameter, Optimizer] = {}
    OptClass = GaLoreAdamW8bit if eightbit else GaLoreAdamW
    for p in model.parameters():
        if p.requires_grad:
            opt = OptClass(
                [{"params": p, "rank": galore_rank, "update_proj_gap": update_proj_gap, "scale": galore_scale, "proj_type": proj_type}],
                lr=lr,
            )
            opt_dict[p] = opt

    def optimizer_hook(param: torch.nn.Parameter):
        if param.grad is None:
            return
        opt = opt_dict[param]
        opt.step()
        opt.zero_grad()

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(optimizer_hook)

    return opt_dict

# -----------------------------
# 評估（可選）
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader, device) -> float:
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss.detach().float()
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))

# -----------------------------
# 主要訓練流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Causal LM with GaLore (memory-efficient).")
    # 基本 I/O
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF 模型路徑或名稱")
    parser.add_argument("--dataset_path", type=str, required=True, help="資料集檔案或資料夾路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出與 checkpoint 目錄")

    # 訓練超參
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3, help="最多保留幾個 checkpoint 目錄")
    parser.add_argument("--seed", type=int, default=42)

    # 精度 / 裝置
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true", help="啟用模型內的 activation/gradient checkpointing")

    # GaLore 相關
    parser.add_argument("--optimizer", type=str,
                        choices=["galore_adamw", "galore_adamw8bit", "galore_adamw8bit_per_layer"],
                        default="galore_adamw")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--galore_scale", type=float, default=0.25)
    parser.add_argument("--update_proj_gap", type=int, default=200)
    parser.add_argument("--proj_type", type=str, default="std", choices=["std", "q", "srht"])

    # 其他
    parser.add_argument("--resume_from", type=str, default=None, help="checkpoint 目錄以恢復訓練")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 隨機種子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 裝置 & dtype
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 模型與 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )

    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.to(device)

    # 資料
    datasets = build_datasets(args.dataset_path, tokenizer, args.block_size)
    train_ds = datasets["train"]
    eval_ds = datasets["validation"] if "validation" in datasets else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=default_data_collator,
        drop_last=True,
    )

    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=default_data_collator,
            drop_last=False,
        )

    # 優化器
    per_layer_mode = (args.optimizer == "galore_adamw8bit_per_layer")
    optimizer: Optional[Optimizer] = None
    per_param_optimizers: Optional[Dict[torch.nn.Parameter, Optimizer]] = None

    if per_layer_mode:
        # 單卡，且不建議搭配 AMP（此腳本會自動關閉 AMP）
        per_param_optimizers = build_per_layer_optimizers(
            model,
            lr=args.learning_rate,
            galore_rank=args.rank,
            galore_scale=args.galore_scale,
            update_proj_gap=args.update_proj_gap,
            proj_type=args.proj_type,
            eightbit=True,
        )
        use_amp = False
    else:
        optimizer = build_optimizer(
            model=model,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            galore_rank=args.rank,
            galore_scale=args.galore_scale,
            update_proj_gap=args.update_proj_gap,
            proj_type=args.proj_type,
            optimizer_type=args.optimizer,
        )
        # AMP 條件
        use_amp = (device.type == "cuda" and dtype in [torch.float16, torch.bfloat16])

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and dtype == torch.float16)

    # 簡單線性 warmup + 常數 LR（你可改成 cosine 等）
    num_update_steps = args.max_steps
    warmup_steps = int(args.warmup_ratio * num_update_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 1.0

    def set_lr(base_lr):
        if optimizer is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr

    # 恢復訓練
    global_step = 0
    best_eval = None
    if args.resume_from:
        ckpt = Path(args.resume_from) / "trainer_state.json"
        if ckpt.exists():
            with open(ckpt, "r", encoding="utf-8") as f:
                state = json.load(f)
            global_step = state.get("global_step", 0)
            best_eval = state.get("best_eval", None)

            # 嘗試載入 model / optimizer
            model.load_state_dict(torch.load(Path(args.resume_from) / "pytorch_model.bin", map_location=device))
            if not per_layer_mode and (Path(args.resume_from) / "optimizer.pt").exists():
                optimizer.load_state_dict(torch.load(Path(args.resume_from) / "optimizer.pt", map_location=device))
            print(f"[Resume] Loaded checkpoint from {args.resume_from} at step {global_step}")

    # 訓練
    model.train()
    progress = tqdm(total=args.max_steps, desc="Training", initial=global_step)

    accum = 0
    running_loss = 0.0
    save_counter = 0

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            lr = args.learning_rate * lr_lambda(global_step)
            set_lr(lr)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16 if dtype == torch.float16 else torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            if use_amp and dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum += 1
            running_loss += loss.item()

            if accum % args.gradient_accumulation_steps == 0:
                if not per_layer_mode:
                    if use_amp and dtype == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # per-layer：更新在 backward hook；這裡只需要把 grads 清乾淨（保險起見）
                    model.zero_grad(set_to_none=True)

                global_step += 1
                progress.update(1)
                progress.set_postfix({"loss": f"{(running_loss):.4f}", "lr": f"{lr:.2e}"})
                running_loss = 0.0

                # 評估
                if eval_loader is not None and args.eval_every > 0 and global_step % args.eval_every == 0:
                    eval_loss = evaluate(model, eval_loader, device)
                    ppl = math.exp(min(eval_loss, 50))
                    tqdm.write(f"[Eval] step={global_step} loss={eval_loss:.4f} ppl={ppl:.2f}")

                    if best_eval is None or eval_loss < best_eval:
                        best_eval = eval_loss
                        save_checkpoint(args.output_dir, model, optimizer, global_step, best_eval, is_best=True, per_layer_mode=per_layer_mode)

                # 儲存 checkpoint
                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(args.output_dir, model, optimizer, global_step, best_eval, per_layer_mode=per_layer_mode)
                    prune_checkpoints(args.output_dir, args.save_total_limit)

        # end for
    # end while

    # 最終儲存
    save_checkpoint(args.output_dir, model, optimizer, global_step, best_eval, per_layer_mode=per_layer_mode, final=True)
    tokenizer.save_pretrained(args.output_dir)
    tqdm.write("Training complete.")


def save_checkpoint(output_dir: str, model, optimizer, step: int, best_eval: Optional[float], per_layer_mode: bool = False, is_best: bool = False, final: bool = False):
    tag = "best" if is_best else ("final" if final else f"step_{step}")
    ckpt_dir = Path(output_dir) / f"checkpoint-{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 儲存 model
    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")

    # 儲存 optimizer（per-layer 模式略過統一 optimizer）
    if (optimizer is not None) and (not per_layer_mode):
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    # trainer_state
    state = {"global_step": step, "best_eval": best_eval}
    with open(ckpt_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # 也同步存一份到 output_dir 根目錄，方便 --resume_from 指定
    torch.save(model.state_dict(), Path(output_dir) / "pytorch_model.bin")
    if (optimizer is not None) and (not per_layer_mode):
        torch.save(optimizer.state_dict(), Path(output_dir) / "optimizer.pt")
    with open(Path(output_dir) / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    tqdm.write(f"[Checkpoint] Saved at {ckpt_dir}")


def prune_checkpoints(output_dir: str, keep: int):
    if keep <= 0:
        return
    root = Path(output_dir)
    ckpts = sorted([p for p in root.glob("checkpoint-*") if p.is_dir()], key=os.path.getmtime, reverse=True)
    for p in ckpts[keep:]:
        try:
            for sub in p.rglob("*"):
                if sub.is_file():
                    sub.unlink()
            p.rmdir()
            tqdm.write(f"[Checkpoint] Pruned {p.name}")
        except Exception as e:
            tqdm.write(f"[Checkpoint] Failed to prune {p.name}: {e}")


if __name__ == "__main__":
    main()
