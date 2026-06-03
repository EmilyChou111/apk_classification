import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from utils import safe_load_bert_tokenizer


class CSVTextDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="bert-base-chinese", max_length=512):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.texts = self.df["text"].tolist()
        self.labels = self.df["sample_state"].tolist()

        self.tokenizer = safe_load_bert_tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class PredictDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="bert-base-chinese", max_length=512):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.texts = self.df["text"].tolist()

        self.tokenizer = safe_load_bert_tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }