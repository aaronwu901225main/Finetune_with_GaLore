import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed, get_constant_schedule
from trl import SFTTrainer, setup_chat_format
from trl.trainer.utils import DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, uuid

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
                # 轉成特殊格式，保留工具呼叫資訊
                parts.append(f"<|im_start|>assistant\n<tool_call>{m['tool_uses']}</tool_call><|im_end|>")
        elif role == "tool":
            if m.get("tool_results"):
                parts.append(f"<|im_start|>tool\n{m['tool_results'][0]}<|im_end|>")

    return {"text": "\n".join(parts)}

# login to huggingface hub
huggingface_hub.login(token="XXX")
# parameters for Galore optimizer
lr = 1e-5
rank = 1024
update_proj_gap = 200
scale = 2
modelpath = "Salesforce/Llama-xLAM-2-8b-fc-r"

set_seed(42)
run_id = f"galore-{str(uuid.uuid4())}"

model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    torch_dtype = torch.bfloat16,
#    attn_implementation = "flash_attention_2",  
    device_map = "auto",
    use_cache = False,
)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

model, tokenizer = setup_chat_format(model, tokenizer)
# 確保有 PAD token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("json", data_files="merged.jsonl")
dataset = dataset["train"].map(format_example)
# 確認處理結果
print(dataset[0]["text"][:500])

training_arguments = TrainingArguments(
    output_dir = f"out_{run_id}",
#    evaluation_strategy = "steps",
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
    optim="galore_adamw8bit_per_layer",
    optim_target_modules=["attn", "mlp"],
    optim_args=f"rank={rank}, update_proj_gap={update_proj_gap}, scale={scale}",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    dataset_text_field="text",
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template = "<|im_start|>user", 
        response_template = "<|im_start|>assistant", 
        tokenizer = tokenizer, 
        mlm = False),
    max_seq_length = 256,
    dataset_kwargs = dict(add_special_tokens = False),
    args = training_arguments,
)
trainer.train()