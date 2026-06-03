import argparse
import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
from utils import safe_load_bert_tokenizer


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def set_chinese_font():
    import matplotlib.font_manager as fm
    for fname in fm.findSystemFonts():
        try:
            prop = fm.FontProperties(fname=fname)
            if "Hei" in prop.get_name() or "YaHei" in prop.get_name() or "Song" in prop.get_name() or "Ming" in prop.get_name():
                plt.rcParams["font.sans-serif"] = [prop.get_name()]
                return
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="数据探索与可视化")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    args = parser.parse_args()

    set_chinese_font()
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    train_df = pd.read_csv(args.train, encoding="utf-8-sig")
    val_df = pd.read_csv(args.val, encoding="utf-8-sig")
    combined = pd.concat([train_df, val_df], ignore_index=True)

    print(f"[explore] 训练集: {len(train_df)}, 验证集: {len(val_df)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    label_counts = combined["sample_state"].value_counts().sort_index()
    labels_names = [idx_to_label.get(k, k) for k in label_counts.index]
    axes[0].bar(labels_names, label_counts.values, color=["#FF6B6B", "#4ECDC4"])
    axes[0].set_title("Category Distribution")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=11)

    text_lengths = combined["text"].apply(lambda t: len(str(t)))
    axes[1].hist(text_lengths, bins=50, color="#45B7D1", edgecolor="white", alpha=0.8)
    axes[1].axvline(text_lengths.mean(), color="red", linestyle="--", label=f"Mean: {text_lengths.mean():.0f}")
    axes[1].axvline(text_lengths.median(), color="orange", linestyle="--", label=f"Median: {text_lengths.median():.0f}")
    axes[1].set_title("Text Length Distribution")
    axes[1].set_xlabel("Character Count")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    tokenizer = safe_load_bert_tokenizer()
    token_lengths = combined["text"].apply(
        lambda t: len(tokenizer.encode(str(t), add_special_tokens=True, truncation=False))
    )
    axes[2].hist(token_lengths, bins=50, color="#96CEB4", edgecolor="white", alpha=0.8)
    axes[2].axvline(512, color="red", linestyle="--", label="max_length=512")
    axes[2].axvline(token_lengths.median(), color="orange", linestyle="--", label=f"Median: {token_lengths.median():.0f}")
    axes[2].set_title("BERT Token Count Distribution")
    axes[2].set_xlabel("Token Count")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "figures", "data_exploration.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[explore] 图表已保存: {fig_path}")

    print(f"\n[explore] 文本长度统计:")
    print(f"  Mean:   {text_lengths.mean():.0f}")
    print(f"  Median: {text_lengths.median():.0f}")
    print(f"  Min:    {text_lengths.min():.0f}")
    print(f"  Max:    {text_lengths.max():.0f}")
    print(f"  >512 chars: {(text_lengths > 512).sum()} ({(text_lengths > 512).mean() * 100:.1f}%)")
    print(f"\n[explore] Token 长度统计:")
    print(f"  Mean:   {token_lengths.mean():.0f}")
    print(f"  Median: {token_lengths.median():.0f}")
    print(f"  >512 tokens: {(token_lengths > 512).sum()} ({(token_lengths > 512).mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()