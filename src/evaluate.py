import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import CSVTextDataset
from model import BertClassifier


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] 混淆矩阵保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="BERT 模型评估")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] 设备: {device}")

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    class_names = list(label_map.keys())

    val_dataset = CSVTextDataset(args.val, args.model_name, args.max_length)
    if args.max_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(args.max_samples, len(val_dataset))))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BertClassifier(args.model_name, num_classes, args.pool)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    per_class_precision = precision_score(all_labels, all_preds, average=None, labels=list(range(num_classes)))
    per_class_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(num_classes)))
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(num_classes)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    print(f"\n{'='*50}")
    print(f"评估结果 (pool={args.pool})")
    print(f"{'='*50}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    print(f"Avg Loss:      {total_loss / len(val_loader):.4f}")
    print(f"\n各类别详细指标:")
    for i, name in enumerate(class_names):
        print(f"  {name}: Precision={per_class_precision[i]:.4f}, Recall={per_class_recall[i]:.4f}, F1={per_class_f1[i]:.4f}")
    print(f"\n混淆矩阵:")
    print(cm)

    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    fig_path = os.path.join(args.output_dir, "figures", f"confusion_matrix_{args.pool}.png")
    plot_confusion_matrix(cm, class_names, fig_path)

    result = {
        "pool": args.pool,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "avg_loss": round(total_loss / len(val_loader), 4),
        "per_class": {
            name: {
                "precision": round(per_class_precision[i], 4),
                "recall": round(per_class_recall[i], 4),
                "f1": round(per_class_f1[i], 4),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }

    result_path = os.path.join(args.output_dir, f"eval_result_{args.pool}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[evaluate] 评估结果保存至: {result_path}")


if __name__ == "__main__":
    main()