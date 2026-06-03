import argparse
import json
import os
import re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SYSTEM_PROMPT = "你是一个APK安全分析助手，只输出类别名称B或W。"


def fuzzy_match(raw_output):
    raw = raw_output.strip().upper()
    if raw in ("B", "W"):
        return raw
    match = re.search(r"\b([BW])\b", raw)
    if match:
        return match.group(1)
    return "B"


def main():
    parser = argparse.ArgumentParser(description="SFT 模型评估与预测")
    parser.add_argument("--adapter", type=str, default="outputs/sft_adapter")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评测样本数")
    parser.add_argument("--output", type=str, default="outputs/llm_sft_results.json")
    parser.add_argument("--predict", type=str, default=None, help="对预测集批量推理")
    parser.add_argument("--predict_output", type=str, default="outputs/predictions_sft.csv")
    args = parser.parse_args()

    LOCAL_QWEN = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2-0.5B-Instruct")
    if os.path.exists(LOCAL_QWEN):
        args.base_model = LOCAL_QWEN

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sft_eval] 设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.merge_and_unload()
    model.eval()

    if args.predict:
        df = pd.read_csv(args.predict, encoding="utf-8-sig")
        if args.max_samples:
            df = df.iloc[:args.max_samples]
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="SFT Predicting"):
            text = str(row.get("text", ""))
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"文本内容：{text}\n类别："},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            pred = fuzzy_match(raw)
            results.append(pred)
        df["predicted_label"] = results
        os.makedirs(os.path.dirname(args.predict_output) if os.path.dirname(args.predict_output) else ".", exist_ok=True)
        df.to_csv(args.predict_output, index=False, encoding="utf-8-sig")
        pred_counts = pd.Series(results).value_counts()
        print(f"[sft_eval] 预测结果保存至: {args.predict_output}")
        print(f"[sft_eval] 预测分布:\n{pred_counts.to_string()}")
        return

    val_df = pd.read_csv(args.val, encoding="utf-8-sig")
    if args.max_samples:
        val_df = val_df.sample(n=args.max_samples, random_state=42)

    true_labels = []
    pred_labels = []
    details = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="SFT Evaluating"):
        text = str(row["text"])
        true_label = idx_to_label.get(int(row["sample_state"]), str(int(row["sample_state"])))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"文本内容：{text}\n类别："},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        pred = fuzzy_match(raw)

        true_labels.append(true_label)
        pred_labels.append(pred)
        details.append({"true": true_label, "pred": pred, "raw_output": raw})

    label_names = list(label_map.keys())
    true_idx = [label_map[t] for t in true_labels]
    pred_idx = [label_map[p] if p in label_map else label_map["B"] for p in pred_labels]

    acc = accuracy_score(true_idx, pred_idx)
    macro_f1 = f1_score(true_idx, pred_idx, average="macro")
    weighted_f1 = f1_score(true_idx, pred_idx, average="weighted")
    per_class_p = precision_score(true_idx, pred_idx, average=None, labels=list(range(len(label_map))))
    per_class_r = recall_score(true_idx, pred_idx, average=None, labels=list(range(len(label_map))))
    per_class_f1_scores = f1_score(true_idx, pred_idx, average=None, labels=list(range(len(label_map))))

    print(f"\n{'='*50}")
    print(f"LLM SFT 评估结果")
    print(f"{'='*50}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    for i, name in enumerate(label_names):
        print(f"  {name}: Precision={per_class_p[i]:.4f}, Recall={per_class_r[i]:.4f}, F1={per_class_f1_scores[i]:.4f}")

    result = {
        "method": "llm_sft",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "samples": len(val_df),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            name: {"precision": round(per_class_p[i], 4), "recall": round(per_class_r[i], 4), "f1": round(per_class_f1_scores[i], 4)}
            for i, name in enumerate(label_names)
        },
        "details": details[:10],
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[sft_eval] 结果保存至: {args.output}")


if __name__ == "__main__":
    main()