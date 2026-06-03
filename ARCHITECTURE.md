
# ARCHITECTURE.md — 技术方案文档

> BERT apk分类项目（结构化 CSV 输入 → 自然语言拼接 → 按 sample_state 划分训练/预测）

---

## 一、项目定位

### 场景选型理由

| 维度 | 说明 |
|------|------|
| 任务选型 | 多分类：从业务 CSV 表格（数十个字段）中，将每一行拼接为自然语言描述，基于 `sample_state=B/W` 行训练分类器，预测 `sample_state=G` 行的类别标签 |
| 数据选型 | 自定义 `input.csv`：包含文本类、数值类、类别型等多列，通过结构化字段拼接形成模型的输入文本，模拟真实业务中“多特征融合为一段描述”的常见需求 |
| 模型选型 | `bert-base-chinese`：参数量适中（110M），可在 CPU 上运行，是学习 fine-tuning 的标准起点；拼接后的文本仍为中文自然语言，适合 BERT 处理 |
| 对比维度 | 三种池化策略 × fine-tuning / SFT / zero-shot LLM，三路对比让教学价值最大化 |

### 三套实现对比

| 维度 | `src/`（BERT fine-tune）| `src_llm/classify_llm.py`（LLM zero-shot）| `src_llm/train_sft.py`（LLM SFT）|
|------|------------------------|------------------------------------------|----------------------------------|
| 目的 | 展示判别式 fine-tuning 全链路 | 展示 zero-shot 分类新范式 | 展示指令微调（SFT）+ LoRA 高效微调 |
| 模型 | bert-base-chinese（110M）| Qwen2-0.5B-Instruct（500M）| Qwen2-0.5B-Instruct（500M + LoRA）|
| 训练 | 需要，约 3~10 min/epoch（GPU）| 不需要训练 | LoRA 3 epoch ≈ 数十分钟（GPU）|
| 可训练参数 | 全量 110M（100%）| 0 | 1.08M（0.22%，LoRA r=8）|
| 准确率（示例）| 视数据而定（例：60%~65%）| 视 prompt 设计，通常低于 fine-tune | 接近甚至超越全量 fine-tune（少量标注时） |
| 代码规模 | ~600 行，5 个模块 | ~90 行，1 个文件 | ~200 行，2 个文件 |
| 教学重点 | 结构化字段拼接 → 自然语言输入；tokenize / pooling / optimizer / loss | prompt 设计 / 输出解析 | chat格式 / loss masking / LoRA 原理 |

---

## 二、整体流水线

```
input.csv（数十个字段 + label + sample_state）
         │
         ▼
  load_data.py              ← 读取 CSV，校验列存在性；构建字段拼接模板；
         │                     将每一行的所有字段（除 label, sample_state）拼接为
         │                     “列名: 值，列名: 值，...” 的自然语言字符串；
         │                     生成 label_map；将 B/W 拆分为 train/val，G 作为预测集
         │
         ▼
  explore_data.py           ← 基于拼接后的文本（B/W 数据）：
         │                     类别分布 / 文本长度（拼接后）/ Token 长度分析（生成图表）
         │
         ▼
  dataset.py                ← CSVTextDataset：根据 split 读取对应 CSV，取拼接好的 text 列进行 tokenize
         │
         ▼
  model.py                  ← BertModel + 池化策略（cls / mean / max）+ Linear
         │
         ▼
  train.py                  ← AdamW（分层 lr）+ warmup + 加权 loss（可选）
         │
         ▼
  evaluate.py               ← accuracy / macro F1 / 混淆矩阵
         │
         ▼
  predict.py                ← 对 G 集批量推理（先拼接字段为文本再编码），
                              合并到原 CSV 输出新 CSV（增加 predicted_label 列）

────── 并行对比 ──────

  src_llm/classify_llm.py   ← Qwen2 zero-shot，同一验证集（或 G 集）对比准确率
         │
         ▼
  src_llm/train_sft.py      ← LoRA 指令微调（chat格式 + loss masking）
         │
         ▼
  src_llm/evaluate_sft.py   ← 加载 adapter，输出预测并追加到 CSV
```

---

## 三、各环节技术选型

### 3.1 数据集：input.csv 自定义数据（含字段优化拼接）

**原始表格结构**：
`input.csv` 包含几十个业务字段，核心列包括：
- `sample_state`：数据划分标识（`B` 已标注训练、`W` 已标注验证、`G` 待预测）
- `label`：类别标签（在 `sample_state=B/W` 中有值，`G` 中可为空）
- 其他大量特征字段，如：
  - 包名类：`apk_md5_new`、`pkg`、`cert_pkg_num`、`cert_appname_max` …
  - 数值类：`device_cnt`、`cert_total_num`、`apk_num` …
  - 日期类：`cert_start_time`、`cert_end_time`、`dt` …
  - 布尔/标志类：`is_w_2`、`eng_has_white`、`is_in_kv`（通常为 Y/N 或 1/0）
  - 文本描述类：`cert_issuer`、`byz_url`、`app_name_max` …

直接将这些原始值喂给模型存在两个问题：
1. 丢失列名语义：模型不知道 `apk_num=1` 表示什么。
2. 格式不自然：日期 `2024/4/2`、布尔值 `Y` 等不符合人类表达习惯，降低预训练语言模型的效果。

**优化后的字段拼接策略**：
在 `load_data.py` 中，对每个特征列进行类型感知的预处理，然后按 `列名: 值` 格式拼接，生成类似人类的描述性文本。

**预处理规则表**：

| 字段类型 | 原始值示例 | 转换后文本 | 说明 |
|---------|------------|-----------|------|
| 普通文本/类别 | `樱花视频` | `樱花视频` | 直接使用，去除首尾空格 |
| 数值 | `36`、`1`、`6736` | `36`、`1`、`6736` | 保留原数值，可考虑加单位（如“个”、“次”）但通常不加，由列名表达 |
| 日期（YYYY/M/D 或 YYYY-MM-DD）| `2024/4/2` | `2024年4月2日` | 转化为中文日期格式，更符合语感 |
| 日期（仅年月日无分隔） | `20240402` | `2024年4月2日` | 需正则解析后转换 |
| 布尔值 Y/N | `Y` | `是` | 统一转为“是/否” |
| 布尔值 1/0 | `1` | `是` | 若字段名暗示布尔含义，统一转换 |
| 缺失值（NaN、空字符串、`\N`）| (空) | `未知` 或 `无记录` | 避免产生空信息，保持文本连贯 |
| 长文本/URL | `yinghua.sagamata.com` | 保留原文 | URL 等直接保留，模型具备一定理解能力 |
| 多值标签（如 `GrayFlag/Android...`） | `GrayFlag/Android.C_SexPlayer.hh[exp];1` | 可替换分隔符为顿号 | 将 `;` 替换为 `、`，使其更像列举 |

**拼接模板**：
```python
def row_to_text(row, feature_columns):
    parts = []
    for col in feature_columns:
        value = row[col]
        # 处理缺失
        if pd.isna(value) or str(value).strip() in ('', '\\N', 'nan'):
            parts.append(f"{col}: 未知")
            continue
        val_str = str(value).strip()
        # 日期转换（示例：2024/4/2 -> 2024年4月2日）
        if re.match(r'\d{4}/\d{1,2}/\d{1,2}', val_str):
            parts.append(f"{col}: {val_str.replace('/', '年',1).replace('/', '月')}日")
        elif re.match(r'\d{4}-\d{1,2}-\d{1,2}', val_str):
            parts.append(f"{col}: {val_str.replace('-', '年',1).replace('-', '月')}日")
        # 布尔转换
        elif val_str.upper() in ('Y', 'YES', '1') and ('has' in col or 'is' in col):
            parts.append(f"{col}: 是")
        elif val_str.upper() in ('N', 'NO', '0') and ('has' in col or 'is' in col):
            parts.append(f"{col}: 否")
        else:
            parts.append(f"{col}: {val_str}")
    return "，".join(parts)
```

**示例**：
原始一行数据（简化）：
| apk_md5_new | device_cnt | cert_appname_max | cert_start_time | eng_has_white | byz_url | label | sample_state |
|-------------|------------|------------------|-----------------|---------------|---------|-------|--------------|
| 04972...    | 1          | 樱花视频          | 2024/4/2        | Y             | yinghua.sagamata.com | 1 | B |

拼接后文本（模型实际输入）：
```
apk_md5_new: 049729F5562CFE71BCF1F77405F6B9D2，device_cnt: 1，cert_appname_max: 樱花视频，cert_start_time: 2024年4月2日，eng_has_white: 是，byz_url: yinghua.sagamata.com，...
```

**选型原因**：
- 自然语言化使预训练模型的语义理解能力最大化，尤其对中文日期、布尔表达的转换能显著降低 token 碎片化。
- “列名: 值”的格式在保持信息完整的同时，将列名本身的语义（如 `device_cnt` 暗示设备数量）注入上下文，相当于给了模型一份数据字典。
- 长度可控：单条记录拼接后通常在 300~800 字之间，配合 `max_length=512` 可覆盖绝大多数样本，极少数超长文本可截断。
- 易于扩展：新增字段只需加入 `feature_columns` 列表，无需改动模型结构。

**划分逻辑**：
- `sample_state ∈ {B, W}` 的行 → 已标注数据集，按 8:2 随机分层拆分为训练集和验证集。
- `sample_state == G` 的行 → 预测集，保留原始顺序，不参与训练。
- `label` 列在 G 行中可能为空，仅用于预测输出时占位。

**后续影响**：
- 数据探索（`explore_data.py`）对拼接后的 `text` 列进行长度和 token 数分析，帮助确定合适的 `max_length` 截断值。
- 所有模型训练和推理均使用拼接后的文本。

---

### 3.2 模型：BertModel + 自定义分类头

**选型原因（不用 BertForSequenceClassification）**：
- `BertForSequenceClassification` 内部结构是黑盒，学生看不到向量提取逻辑。
- 手写分类头只有 3 行核心代码，池化策略替换清晰可见。
- 方便后续扩展：换 RoBERTa、加多任务头、换 pooling 策略都无需改框架代码。

**三种池化策略**：

| 策略 | 实现 | 直觉解释 | 适用场景 |
|------|------|---------|---------|
| `cls` | `last_hidden[:, 0, :]` | BERT 训练时 [CLS] 就被设计为句子摘要向量 | 分类任务的默认选择 |
| `mean` | 有效 token 均值（排除 padding） | 所有词信息的平均表达 | 语义相似度、句子表示 |
| `max` | 有效 token 逐维取最大值 | 保留每个维度最显著的激活 | 情感类、关键词驱动任务 |

> **教学设计**：三种策略通过 `--pool cls/mean/max` 切换，训练结果存在不同 checkpoint，
> 最后用 evaluate.py 对比混淆矩阵，让学生量化差异。

### 3.3 优化器：分层学习率

```
BERT 层：    lr = 2e-5   （预训练权重，小步微调）
分类头：     lr = 1e-4   （随机初始化，需要更快收敛）
```

**选型原因**：
- BERT 预训练权重已经很好，用过大的学习率会 "遗忘" 预训练知识（catastrophic forgetting）
- 分类头是新加的随机初始化层，需要更大步长才能快速学到映射关系
- 统一用 AdamW（带 weight decay）是 Transformer fine-tuning 的事实标准

### 3.4 类别不均衡：加权 CrossEntropyLoss

```python
# sklearn 计算 balanced weight：
# weight_i = n_samples / (n_classes × n_samples_i)
weights = compute_class_weight("balanced", classes=classes, y=labels)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
```

**选型原因**：
- 若类别分布不均衡，balanced weight 让每个类别对 loss 的贡献大致相等，提升小类 Recall。
- 通过 `--use_class_weight` 开关，学生可对比加权前后小类别性能变化。

### 3.5 学习率调度：Linear Warmup + Decay

- 使用 HuggingFace `get_linear_schedule_with_warmup`，warmup 比例默认 0.1。
- 前 10% 步数学习率从 0 线性升至设定值，之后线性衰减至 0。
- 防止训练初期破坏预训练权重，后期精细调优。

---

### 3.6 LLM SFT：LoRA 指令微调

**核心教学点**：

#### 数据格式转换

将分类任务转化为 chat 格式。从训练集 CSV 中，每一行已有拼接好的文本和标签，构造：

```
system:    "你是一个文本分类助手，只输出类别名称，可选：类别A/类别B/..."
user:      "文本内容：{拼接后的文本}\n类别："
assistant: "{label_name}"         ← 只有这个 token 参与 loss 计算
```

#### Loss Masking（SFT 与 Pretraining 的核心区别）

```python
# 只在 assistant 回复部分（类别名 + EOS）计算 loss
labels = [-100] * prompt_len + response_ids
```

#### LoRA（Low-Rank Adaptation）

```
原始矩阵 W ∈ R^{d×d}（冻结）
         ↓
输出 += B·A·x，其中 A ∈ R^{r×d}，B ∈ R^{d×r}，r=8 << d=896

Qwen2-0.5B 全参数：495,114,112
LoRA r=8 可训练参数：1,081,344（0.22%）
```

**选型原因**：
- 显存友好：RTX 4060 8GB 可跑，全量微调会 OOM。
- 速度快：仅训练 0.22% 参数，3 epoch 约数十分钟。
- 业界 SFT 标准实践，可迁移至其他任务。

#### 实测结果（以实际数据为准）

| 配置 | 数据量 | 训练时间（GPU） | 可训练参数 | 验证准确率 |
|------|--------|----------------|-----------|------------|
| LoRA r=8，3 epoch | B/W 训练集 | ~15-30 min | 1.08M（0.22%）| 待实测 |

---

## 四、评估体系

### 4.1 核心指标

| 指标 | 含义 | 为什么用 |
|------|------|---------|
| Accuracy | 正确预测数 / 总数 | 直觉易懂，但不均衡时会高估 |
| Macro F1 | 各类 F1 的算术平均 | 每个类等权重，能反映小类的真实性能 |
| Per-class Precision/Recall | 每个类别单独统计 | 定位问题：哪个类被预测错了？被误判成了哪类？ |
| 混淆矩阵 | 真实类别 × 预测类别 | 可视化易混淆的类别对 |

### 4.2 消融实验矩阵

通过改变单一变量，量化每个设计决策的价值：

| 实验 | 变量 | 对比维度 |
|------|------|---------|
| 池化策略 | `--pool cls/mean/max` | 三种向量提取方式的精度差异 |
| 类别加权 | `--use_class_weight` | 加权前后小类 Recall 变化 |
| 训练轮数 | `--epochs 1/3/5` | epoch 数对收敛和过拟合的影响 |
| 截断长度 | `--max_length 128/256/512` | 拼接后文本可能较长，需要探索最佳截断 |
| 数据量 | 取训练集子集（1k/5k/全量） | 少样本下 LLM zero-shot 是否更有优势 |

### 4.3 三方方案对比（以实际数据为准）

| 对比维度 | BERT fine-tune | LLM Zero-shot | LLM SFT（LoRA）|
|---------|---------------|---------------|----------------|
| 模型 | bert-base-chinese（110M）| Qwen2-0.5B（500M）| Qwen2-0.5B（500M）|
| 可训练参数 | 110M（全量）| 0 | 1.08M（0.22%）|
| 训练成本 | 3 epoch，GPU 数十分钟 | 无 | 3 epoch，GPU 约 20 min |
| 推理速度 | ~5ms/条（GPU）| ~2s/条（CPU）| ~0.06s/条（GPU）|
| 准确率 | 视数据而定 | 视 prompt 设计 | 视数据量/质量 |
| 数据需求 | 需要较多标注数据 | 零标注 | 少量标注即可见效 |
| 输出解析失败 | 无（直接 logits）| 可能存在（模糊匹配处理）| 可能存在（模糊匹配处理）|
| 适用场景 | 标注数据充足，追求效率 | 快速冷启动、无标注 | 少量标注 + 大模型能力 |

---

## 五、关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| CSV 读取编码错误 | 文件使用 GBK/BOM 等编码 | 先用 `chardet` 检测编码，或尝试 `utf-8-sig`、`gbk` 兜底 |
| `label` 列在 G 样本中为空 | 缺少标签导致训练/预测逻辑报错 | 在数据加载时过滤 G 行，预测时不读取 label |
| 类别标签非数字或含空格 | 导致模型输出与标签匹配困难 | 统一构建 `label_map.json`，将原始标签映射为整数 ID |
| Windows 多进程 DataLoader 报错 | PyTorch 多进程 pickling 问题 | `num_workers=0`，Linux 可设 2/4 |
| 单条推理时 Tensor 维度错误 | `return_tensors="pt"` 多出 batch 维度 | `encoding["input_ids"].squeeze(0)` |
| 预测结果写回 CSV 时列错位 | 原始 CSV 可能有索引或空行 | 使用 `pandas` 按行索引合并，`reset_index(drop=True)` |
| Qwen2 输出包含额外文字 | CausalLM 生成不严格受控 | 使用模糊匹配 `if class_name in raw_output` |
| SFT 时 `apply_chat_template` 返回 BatchEncoding | transformers 版本差异 | 先用 `tokenize=False` 获得文本，再 `tokenizer.encode()` |
| LoRA 模型加载后推理速度慢 | 未合并权重 | 可选择 `model.merge_and_unload()` 加快推理 |
| matplotlib 中文乱码 | 默认字体不含 CJK 字符 | 动态检测系统中文字体，失败则使用英文标签 |
| 拼接后文本过长，超出 BERT 512 限制 | 字段过多或值较长 | 设置 `max_length=512`，通过 `truncation=True` 截断；数据探索时给出长度分布，可考虑按重要性排序裁剪字段 |
| 字段值包含特殊符号（如 “:”、“，”）干扰拼接格式 | 可能与分隔符冲突 | 预处理时对值中的分隔符进行转义或替换；或使用 `\n` 换行分隔，“列名: 值”保持清晰 |
| 特征列中包含大量缺失值 | 拼接后出现大量“未知”，降低信息密度 | 数据探索时统计缺失率，考虑在拼接前丢弃缺失率过高的列，或保留但调整表达方式（如“该字段无记录”） |
| 日期格式不统一（如 2024/4/2 与 20240402） | 下游处理无法解析 | 在拼接前用统一的正则/日期解析函数转换为标准中文日期 |
| 布尔字段语义模糊（Y/N 与 1/0 混用） | 模型难以学习 | 根据字段名关键词（has/is）自动判断并统一转换为“是/否” |

---

## 六、目录结构

```
bert_text_classification_csv/
│
├── src/                          # fine-tuning 实现
│   ├── load_data.py              # 读取 input.csv，字段拼接为自然语言；生成 label_map；划分 train/val/predict 集
│   ├── explore_data.py           # 数据探索（类别分布、拼接文本长度、Token 长度分析，基于 B/W）
│   ├── dataset.py                # CSVTextDataset：加载 CSV，取拼接好的 text 列进行 tokenize
│   ├── model.py                  # BertClassifier（cls / mean / max 池化）
│   ├── train.py                  # 训练循环（分层 lr、warmup、加权 loss、checkpoint）
│   ├── evaluate.py               # 评估 + 混淆矩阵可视化
│   └── predict.py                # 对 G 集批量推理（先拼接字段为文本再编码），输出带预测标签的完整 CSV
│
├── src_llm/
│   ├── classify_llm.py           # Qwen2-0.5B zero-shot 分类（使用同一 label_map）
│   ├── train_sft.py              # LoRA 指令微调（chat 格式 + loss masking）
│   └── evaluate_sft.py           # 加载 adapter，对 G 集预测并追加到输出 CSV
│
├── data/                         # 输入与中间数据
│   ├── input.csv                 # 原始输入文件（含 B/W/G 及所有特征列）
│   ├── train.csv                 # 由 B/W 拆分出的训练集（已包含拼接的 text 列）
│   ├── val.csv                   # 验证集（已包含拼接的 text 列）
│   └── label_map.json            # 标签到 id 的映射
│
├── outputs/
│   ├── checkpoints/              # best_{pool}.pt
│   ├── figures/                  # 数据分析图表
│   ├── train_log_{pool}.json     # 每 epoch 的 loss / acc
│   ├── predictions.csv           # 最终输出：原始 input.csv 所有行（含 G） + predicted_label 列
│   ├── llm_zero_shot_results.json
│   ├── llm_sft_results.json
│   ├── train_log_sft.json
│   └── sft_adapter/              # LoRA adapter 权重与配置
│
├── ARCHITECTURE.md               # 本文件
├── USAGE_GUIDE.md
├── RESUME_GUIDE.md
└── requirements.txt
```