import argparse
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


SYSTEM_PROMPT = "你是一个APK安全分析助手，只输出类别名称B或W。"


def build_chat_data(row_text, label_str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"文本内容：{row_text}\n类别："},
        {"role": "assistant", "content": label_str},
    ]
    return messages


class SFTChatDataset(Dataset):
    def __init__(self, csv_path, label_map, tokenizer, max_length=768):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])
        label_id = int(self.df.iloc[idx]["sample_state"])
        label_str = self.idx_to_label[label_id]

        messages = build_chat_data(text, label_str)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        full_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        response_template = "<|im_start|>assistant\n"
        response_ids = self.tokenizer.encode(response_template + label_str + self.tokenizer.eos_token, add_special_tokens=False)

        input_ids = full_encoding["input_ids"][0]
        labels = input_ids.clone()

        response_start = len(input_ids) - len(response_ids)
        if response_start < 0:
            response_start = 0
        labels[:response_start] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_encoding["attention_mask"][0],
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT 指令微调")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_adapter")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    args = parser.parse_args()

    LOCAL_QWEN = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2-0.5B-Instruct")
    if os.path.exists(LOCAL_QWEN):
        args.model_name = LOCAL_QWEN

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sft] 设备: {device}")

    print(f"[sft] 加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SFTChatDataset(args.train, label_map, tokenizer, args.max_length)
    val_dataset = SFTChatDataset(args.val, label_map, tokenizer, args.max_length)

    if args.max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    train_log = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        log_entry = {"epoch": epoch, "train_loss": round(avg_train_loss, 4), "val_loss": round(avg_val_loss, 4)}
        train_log.append(log_entry)

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> 保存最佳模型")

    log_path = os.path.join(args.output_dir, "train_log_sft.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"logs": train_log, "best_val_loss": best_loss}, f, ensure_ascii=False, indent=2)
    print(f"\n[sft] 训练日志: {log_path}")


if __name__ == "__main__":
    main()