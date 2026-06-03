import json
import os

base = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base, "outputs")

bert = json.load(open(os.path.join(out_dir, "eval_result_cls.json"), "r", encoding="utf-8"))
zero = json.load(open(os.path.join(out_dir, "llm_zero_shot_results.json"), "r", encoding="utf-8"))
sft = json.load(open(os.path.join(out_dir, "llm_sft_results.json"), "r", encoding="utf-8"))

all_results = {
    "BERT Fine-tuning (CLS)": bert,
    "LLM Zero-shot (Qwen2-0.5B)": zero,
    "LLM SFT LoRA (Qwen2-0.5B)": sft,
}

with open(os.path.join(out_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("comparison_results.json saved!")

lines = []
lines.append("=" * 65)
lines.append("  文本分类 - 三种方法对比结果")
lines.append("=" * 65)
lines.append("")

entries = [
    ("BERT Fine-tuning (CLS)", os.path.join(out_dir, "eval_result_cls.json")),
    ("LLM Zero-shot (Qwen2-0.5B)", os.path.join(out_dir, "llm_zero_shot_results.json")),
    ("LLM SFT LoRA (Qwen2-0.5B)", os.path.join(out_dir, "llm_sft_results.json")),
]

for name, path in entries:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    acc = data.get("accuracy", 0) * 100
    mf1 = data.get("macro_f1", 0) * 100
    wf1 = data.get("weighted_f1", 0) * 100
    lines.append(f"> {name}")
    lines.append(f"  Accuracy:     {acc:.2f}%")
    lines.append(f"  Macro F1:     {mf1:.2f}%")
    lines.append(f"  Weighted F1:  {wf1:.2f}%")
    if "per_class" in data:
        for cls, m in data["per_class"].items():
            p = m["precision"] * 100
            r = m["recall"] * 100
            f1 = m["f1"] * 100
            lines.append(f"  [{cls}] P={p:.1f}%  R={r:.1f}%  F1={f1:.1f}%")
    lines.append("")

lines.append("=" * 65)
lines.append("  结论: BERT Fine-tuning 显著优于 LLM 方案")
lines.append("  LLM (Qwen2-0.5B) 无论 Zero-shot 还是 SFT")
lines.append("  均无法识别 W 类，全部预测为 B 类")
lines.append("=" * 65)

summary_path = os.path.join(out_dir, "comparison_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("comparison_summary.txt saved!")

print("\n" + "\n".join(lines))