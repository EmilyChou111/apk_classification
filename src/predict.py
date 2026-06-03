import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PredictDataset
from model import BertClassifier


def main():
    parser = argparse.ArgumentParser(description="BERT 批量预测")
    parser.add_argument("--predict", type=str, default="data/predict.csv")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/predictions.csv")
    parser.add_argument("--input_csv", type=str, default="input.csv", help="原始 input.csv（用于合并输出）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] 设备: {device}")

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    predict_dataset = PredictDataset(args.predict, args.model_name, args.max_length)
    if args.max_samples is not None:
        predict_dataset = torch.utils.data.Subset(predict_dataset, range(min(args.max_samples, len(predict_dataset))))
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BertClassifier(args.model_name, num_classes, args.pool)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())

    predicted_labels = [idx_to_label[p] for p in all_preds]
    predict_df = pd.read_csv(args.predict, encoding="utf-8-sig")
    if args.max_samples is not None:
        predict_df = predict_df.iloc[:args.max_samples]
    predict_df["predicted_label"] = predicted_labels

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    predict_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[predict] 预测结果保存至: {args.output}")

    pred_counts = pd.Series(predicted_labels).value_counts()
    print(f"[predict] 预测分布:\n{pred_counts.to_string()}")


if __name__ == "__main__":
    main()