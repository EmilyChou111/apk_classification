import argparse
import json
import os
import re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SYSTEM_PROMPT = """你是一个专业的APK安全分析助手。请根据提供的APK特征信息，判断该APK的安全等级分类。

分类规则：
- B (Black/黑样本): 恶意或高风险APK，包含恶意软件特征、可疑行为或欺诈特征
- W (White/白样本): 安全或低风险APK，行为正常可信

请仔细分析提供的APK特征信息，仅输出一个字母：B 或 W，不要输出任何其他内容。"""


def build_user_prompt(text):
    return f"请分析以下APK的特征信息，判断其安全等级：\n\n{text}\n\n分类结果（仅输出B或W）："


def fuzzy_match(raw_output):
    raw = raw_output.strip().upper()
    if raw in ("B", "BLACK", "黑"):
        return "B"
    elif raw in ("W", "WHITE", "白"):
        return "W"

    match = re.search(r"\b([BW])\b", raw)
    if match:
        return match.group(1)

    if "黑" in raw or "BLACK" in raw or "恶意" in raw or "高" in raw or "危险" in raw:
        return "B"
    if "白" in raw or "WHITE" in raw or "安全" in raw or "低" in raw or "正常" in raw:
        return "W"

    return None


def main():
    parser = argparse.ArgumentParser(description="LLM Zero-shot 分类")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评测样本数")
    parser.add_argument("--output", type=str, default="outputs/llm_zero_shot_results.json")
    parser.add_argument("--predict", type=str, default=None, help="对预测集批量推理")
    parser.add_argument("--predict_output", type=str, default="outputs/predictions_llm_zero.csv")
    args = parser.parse_args()

    LOCAL_QWEN = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2-0.5B-Instruct")
    if os.path.exists(LOCAL_QWEN):
        args.model_name = LOCAL_QWEN

    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[llm_zero] 加载模型: {args.model_name}, 设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype="auto"
    ).to(device)
    model.eval()

    if args.predict:
        df = pd.read_csv(args.predict, encoding="utf-8-sig")
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot Predicting"):
            text = str(row.get("text", ""))
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(text)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            pred = fuzzy_match(raw)
            if pred is None:
                pred = "B"
            results.append(pred)
        df["predicted_label"] = results
        os.makedirs(os.path.dirname(args.predict_output) if os.path.dirname(args.predict_output) else ".", exist_ok=True)
        df.to_csv(args.predict_output, index=False, encoding="utf-8-sig")
        pred_counts = pd.Series(results).value_counts()
        print(f"[llm_zero] 预测结果保存至: {args.predict_output}")
        print(f"[llm_zero] 预测分布:\n{pred_counts.to_string()}")
        return

    val_df = pd.read_csv(args.val, encoding="utf-8-sig")
    if args.max_samples:
        val_df = val_df.sample(n=args.max_samples, random_state=42)

    true_labels = val_df["sample_state"].tolist()
    pred_labels = []
    details = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Zero-shot Evaluating"):
        text = str(row["text"])
        true_label = idx_to_label.get(int(row["sample_state"]), str(int(row["sample_state"])))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        pred = fuzzy_match(raw)

        if pred is None:
            pred = "B"

        pred_labels.append(pred)
        details.append({"true": true_label, "pred": pred, "raw_output": raw})

    label_names = list(label_map.keys())
    true_idx = [int(t) for t in true_labels]
    pred_idx = [label_map[p] if p in label_map else label_map["B"] for p in pred_labels]

    acc = accuracy_score(true_idx, pred_idx)
    macro_f1 = f1_score(true_idx, pred_idx, average="macro")
    weighted_f1 = f1_score(true_idx, pred_idx, average="weighted")
    per_class_p = precision_score(true_idx, pred_idx, average=None, labels=list(range(len(label_map))))
    per_class_r = recall_score(true_idx, pred_idx, average=None, labels=list(range(len(label_map))))
    per_class_f1_scores = f1_score(true_idx, pred_idx, average=None, labels=list(range(len(label_map))))

    print(f"\n{'='*50}")
    print(f"LLM Zero-shot 评估结果")
    print(f"{'='*50}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    for i, name in enumerate(label_names):
        print(f"  {name}: Precision={per_class_p[i]:.4f}, Recall={per_class_r[i]:.4f}, F1={per_class_f1_scores[i]:.4f}")

    fail_count = sum(1 for d in details if d["pred"] is None)
    print(f"\n解析失败率: {fail_count}/{len(details)} ({fail_count / len(details) * 100:.1f}%)")

    result = {
        "method": "llm_zero_shot",
        "model": args.model_name,
        "samples": len(val_df),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            name: {"precision": round(per_class_p[i], 4), "recall": round(per_class_r[i], 4), "f1": round(per_class_f1_scores[i], 4)}
            for i, name in enumerate(label_names)
        },
        "parse_fail_rate": round(fail_count / len(details), 4),
        "details": details[:10],
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[llm_zero] 结果保存至: {args.output}")


if __name__ == "__main__":
    main()