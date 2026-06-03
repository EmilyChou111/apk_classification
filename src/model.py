import torch
import torch.nn as nn
from utils import safe_load_bert_model


class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_classes=2, pool_strategy="cls", dropout=0.1):
        super().__init__()
        self.bert = safe_load_bert_model()
        self.pool_strategy = pool_strategy
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _pool(self, last_hidden, attention_mask):
        if self.pool_strategy == "cls":
            return last_hidden[:, 0, :]

        if self.pool_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        if self.pool_strategy == "max":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            last_hidden_masked = last_hidden.clone()
            last_hidden_masked[mask_expanded == 0] = -1e9
            return torch.max(last_hidden_masked, dim=1).values

        raise ValueError(f"不支持的池化策略: {self.pool_strategy}，可选: cls/mean/max")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits