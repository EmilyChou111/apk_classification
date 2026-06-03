import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import CSVTextDataset
from model import BertClassifier


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(dataloader), correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="BERT 文本分类训练")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr_bert", type=float, default=2e-5)
    parser.add_argument("--lr_cls", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 设备: {device}")

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    train_dataset = CSVTextDataset(args.train, args.model_name, args.max_length)
    val_dataset = CSVTextDataset(args.val, args.model_name, args.max_length)

    if args.max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BertClassifier(args.model_name, num_classes, args.pool, args.dropout)
    model.to(device)

    bert_params = model.bert.parameters()
    cls_params = model.classifier.parameters()
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": args.lr_bert},
        {"params": cls_params, "lr": args.lr_cls},
    ])

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if args.use_class_weight:
        all_labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
        classes = sorted(set(all_labels))
        weights = compute_class_weight("balanced", classes=np.array(classes), y=all_labels)
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"[train] 类别权重(balanced): {dict(zip(classes, weights))}")
    else:
        criterion = nn.CrossEntropyLoss()

    train_log = []
    best_acc = 0.0
    best_epoch = 0
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"best_{args.pool}.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        train_log.append(log_entry)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "pool": args.pool,
                "label_map": label_map,
                "idx_to_label": idx_to_label,
                "max_length": args.max_length,
            }, checkpoint_path)
            print(f"  -> 保存最佳模型 (Acc: {best_acc:.4f})")

    log_path = os.path.join(args.output_dir, f"train_log_{args.pool}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"pool": args.pool, "logs": train_log, "best_epoch": best_epoch, "best_acc": best_acc}, f, ensure_ascii=False, indent=2)
    print(f"\n[train] 训练日志: {log_path}")
    print(f"[train] 最佳准确率: {best_acc:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()