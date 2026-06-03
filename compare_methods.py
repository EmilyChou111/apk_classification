import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f">>> 命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"[警告] 命令返回非零退出码: {result.returncode}")
    return result


def run_step1_data_prep(args):
    print("\n" + "=" * 60)
    print(">>> 步骤 1/6: 数据加载与字段拼接")
    print("=" * 60)
    cmd = [
        sys.executable, os.path.join("src", "load_data.py"),
        "--input", args.input,
        "--output_dir", args.output_data,
        "--val_ratio", str(args.val_ratio),
    ]
    run_cmd(cmd, "load_data.py")


def run_step2_explore(args):
    print("\n" + "=" * 60)
    print(">>> 步骤 2/6: 数据探索与可视化")
    print("=" * 60)
    cmd = [
        sys.executable, os.path.join("src", "explore_data.py"),
        "--train", os.path.join(args.output_data, "train.csv"),
        "--val", os.path.join(args.output_data, "val.csv"),
        "--label_map", os.path.join(args.output_data, "label_map.json"),
    ]
    run_cmd(cmd, "explore_data.py")


def run_step3_bert_train(args):
    print("\n" + "=" * 60)
    print(">>> 步骤 3/6: BERT Fine-tuning（三种池化策略）")
    print("=" * 60)
    pooling_strategies = ["cls", "mean", "max"]

    for pool in pooling_strategies:
        label = f"BERT fine-tune ({pool} pooling)"
        cmd = [
            sys.executable, os.path.join("src", "train.py"),
            "--train", os.path.join(args.output_data, "train.csv"),
            "--val", os.path.join(args.output_data, "val.csv"),
            "--label_map", os.path.join(args.output_data, "label_map.json"),
            "--pool", pool,
            "--epochs", str(args.bert_epochs),
            "--batch_size", str(args.bert_batch_size),
            "--max_length", str(args.max_length),
            "--output_dir", args.output_dir,
            "--seed", str(args.seed),
        ]
        if args.use_class_weight:
            cmd.append("--use_class_weight")
        run_cmd(cmd, label)


def run_step4_bert_eval(args):
    print("\n" + "=" * 60)
    print(">>> 步骤 4/6: BERT 评估（三种池化策略）")
    print("=" * 60)
    pooling_strategies = ["cls", "mean", "max"]
    results = {}

    for pool in pooling_strategies:
        label = f"BERT eval ({pool} pooling)"
        cmd = [
            sys.executable, os.path.join("src", "evaluate.py"),
            "--val", os.path.join(args.output_data, "val.csv"),
            "--label_map", os.path.join(args.output_data, "label_map.json"),
            "--checkpoint", os.path.join(args.output_dir, "checkpoints", f"best_{pool}.pt"),
            "--pool", pool,
            "--max_length", str(args.max_length),
            "--output_dir", args.output_dir,
        ]
        run_cmd(cmd, label)

        result_path = os.path.join(args.output_dir, f"eval_result_{pool}.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results[pool] = json.load(f)

    return results


def run_step5_bert_predict(args, pool="cls"):
    print("\n" + "=" * 60)
    print(f">>> 步骤 5/6: BERT 预测 G 集 (pool={pool})")
    print("=" * 60)
    cmd = [
        sys.executable, os.path.join("src", "predict.py"),
        "--predict", os.path.join(args.output_data, "predict.csv"),
        "--checkpoint", os.path.join(args.output_dir, "checkpoints", f"best_{pool}.pt"),
        "--label_map", os.path.join(args.output_data, "label_map.json"),
        "--pool", pool,
        "--max_length", str(args.max_length),
        "--output", os.path.join(args.output_dir, "predictions.csv"),
    ]
    run_cmd(cmd, f"BERT predict ({pool})")


def run_step6_llm_zero_shot(args):
    print("\n" + "=" * 60)
    print(">>> 步骤 6a: LLM Zero-shot 分类")
    print("=" * 60)
    cmd = [
        sys.executable, os.path.join("src_llm", "classify_llm.py"),
        "--val", os.path.join(args.output_data, "val.csv"),
        "--label_map", os.path.join(args.output_data, "label_map.json"),
        "--max_samples", str(args.llm_max_samples) if args.llm_max_samples else "None",
        "--output", os.path.join(args.output_dir, "llm_zero_shot_results.json"),
    ]
    run_cmd(cmd, "LLM Zero-shot")


def run_step6_llm_sft_train(args):
    print("\n" + "=" * 60)
    print(">>> 步骤 6b: LLM SFT LoRA 微调")
    print("=" * 60)
    cmd = [
        sys.executable, os.path.join("src_llm", "train_sft.py"),
        "--train", os.path.join(args.output_data, "train.csv"),
        "--val", os.path.join(args.output_data, "val.csv"),
        "--label_map", os.path.join(args.output_data, "label_map.json"),
        "--output_dir", os.path.join(args.output_dir, "sft_adapter"),
        "--epochs", str(args.sft_epochs),
        "--batch_size", str(args.sft_batch_size),
        "--max_length", str(args.sft_max_length),
        "--lora_r", str(args.lora_r),
    ]
    run_cmd(cmd, "LLM SFT training")


def run_step6_llm_sft_eval(args):
    print("\n" + "=" * 60)
    print(">>> 步骤 6c: LLM SFT 评估")
    print("=" * 60)
    cmd = [
        sys.executable, os.path.join("src_llm", "evaluate_sft.py"),
        "--adapter", os.path.join(args.output_dir, "sft_adapter"),
        "--val", os.path.join(args.output_data, "val.csv"),
        "--label_map", os.path.join(args.output_data, "label_map.json"),
        "--max_samples", str(args.llm_max_samples) if args.llm_max_samples else "None",
        "--output", os.path.join(args.output_dir, "llm_sft_results.json"),
    ]
    run_cmd(cmd, "LLM SFT eval")


def plot_comparison(all_results, args):
    methods = []
    accuracies = []
    macro_f1s = []

    for key, result in all_results.items():
        methods.append(key)
        accuracies.append(result.get("accuracy", 0))
        macro_f1s.append(result.get("macro_f1", 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    bar_colors = colors[:len(methods)]

    x = np.arange(len(methods))
    width = 0.35

    axes[0].bar(x, accuracies, width, color=bar_colors, edgecolor="white")
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha="right")
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(x, macro_f1s, width, color=bar_colors, edgecolor="white")
    axes[1].set_title("Macro F1 Comparison")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha="right")
    for i, v in enumerate(macro_f1s):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "figures", "method_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n[comparison] 对比图保存至: {fig_path}")


def print_comparison_table(all_results):
    print(f"\n{'='*80}")
    print(f"文本分类方法对比结果汇总")
    print(f"{'='*80}")

    headers = ["方法", "Accuracy", "Macro F1", "Weighted F1"]
    rows = []
    for key, result in all_results.items():
        rows.append([
            key,
            f"{result.get('accuracy', 0):.4f}",
            f"{result.get('macro_f1', 0):.4f}",
            f"{result.get('weighted_f1', 0):.4f}",
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]
    fmt = "|".join(f" {{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + len(headers) - 1))
    for row in rows:
        print(fmt.format(*row))

    print()

    print(f"{'='*80}")
    print(f"各类别详细指标对比")
    print(f"{'='*80}")

    sub_headers = ["方法", "类别", "Precision", "Recall", "F1"]
    sub_rows = []
    for key, result in all_results.items():
        per_class = result.get("per_class", {})
        for cls_name, metrics in per_class.items():
            sub_rows.append([
                key,
                cls_name,
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('f1', 0):.4f}",
            ])

    sub_col_widths = [max(len(str(r[i])) for r in [sub_headers] + sub_rows) + 2 for i in range(len(sub_headers))]
    sub_fmt = "|".join(f" {{:<{w}}}" for w in sub_col_widths)
    print(sub_fmt.format(*sub_headers))
    print("-" * (sum(sub_col_widths) + len(sub_headers) - 1))
    for row in sub_rows:
        print(sub_fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(description="文本分类方法对比实验")
    parser.add_argument("--input", type=str, default="input.csv")
    parser.add_argument("--output_data", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--bert_epochs", type=int, default=3)
    parser.add_argument("--bert_batch_size", type=int, default=8)
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_batch_size", type=int, default=2)
    parser.add_argument("--sft_max_length", type=int, default=768)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--llm_max_samples", type=int, default=100, help="LLM评测最多样本数")
    parser.add_argument("--skip_bert", action="store_true", help="跳过 BERT 训练")
    parser.add_argument("--skip_llm_zero", action="store_true", help="跳过 LLM Zero-shot")
    parser.add_argument("--skip_llm_sft", action="store_true", help="跳过 LLM SFT")
    parser.add_argument("--skip_explore", action="store_true", help="跳过数据探索")
    parser.add_argument("--bert_only", action="store_true", help="仅运行 BERT")
    parser.add_argument("--llm_only", action="store_true", help="仅运行 LLM")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    t_start = time.time()

    run_step1_data_prep(args)

    if not args.skip_explore:
        run_step2_explore(args)

    if not args.skip_bert and not args.llm_only:
        run_step3_bert_train(args)

    all_results = {}

    if not args.skip_bert and not args.llm_only:
        bert_results = run_step4_bert_eval(args)
        for pool, result in bert_results.items():
            all_results[f"BERT-{pool}"] = result
        run_step5_bert_predict(args, pool="cls")

    if not args.skip_llm_zero and not args.bert_only:
        run_step6_llm_zero_shot(args)
        llm_zero_path = os.path.join(args.output_dir, "llm_zero_shot_results.json")
        if os.path.exists(llm_zero_path):
            with open(llm_zero_path, "r", encoding="utf-8") as f:
                all_results["LLM Zero-shot"] = json.load(f)

    if not args.skip_llm_sft and not args.bert_only:
        run_step6_llm_sft_train(args)
        run_step6_llm_sft_eval(args)
        llm_sft_path = os.path.join(args.output_dir, "llm_sft_results.json")
        if os.path.exists(llm_sft_path):
            with open(llm_sft_path, "r", encoding="utf-8") as f:
                all_results["LLM SFT (LoRA)"] = json.load(f)

    t_end = time.time()

    print_comparison_table(all_results)

    if len(all_results) > 1:
        plot_comparison(all_results, args)

    comparison_path = os.path.join(args.output_dir, "comparison_results.json")
    all_results["_total_time_seconds"] = round(t_end - t_start, 1)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[comparison] 对比结果保存至: {comparison_path}")
    print(f"[comparison] 总耗时: {t_end - t_start:.1f}s")

    return all_results


if __name__ == "__main__":
    main()