# USAGE_GUIDE.md

## 环境准备

### 1. 创建虚拟环境（推荐）
```bash
conda create -n bert_csv python=3.10 -y
conda activate bert_csv
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 检查 GPU（可选）
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
若输出 `True`，训练和推理将自动使用 GPU 加速。

---

## 数据准备

### 输入文件格式
将待分析的数据保存为 `data/input.csv`，必须包含以下列：
- `sample_state`：`B` 或 `W`（已标注训练/验证），`G`（待预测）
- `label`：类别标签（`B`/`W` 行有值，`G` 行可为空）
- 其他任意数量的特征列（数值、文本、日期等）

示例（仅展示关键列）：
```
sample_state,apk_md5_new,device_cnt,cert_appname_max,cert_start_time,eng_has_white,byz_url,label
B,049729F5...,1,樱花视频,2024/4/2,Y,yinghua.sagamata.com,1
W,0A23BCF1...,3,计算器,2023/11/15,N,calc.example.com,0
G,...........,5,未知应用,,,,
```

### 一键运行完整流程
```bash
# 1. 数据加载与拼接（生成 train.csv / val.csv / label_map.json）
python src/load_data.py --input data/input.csv --output_dir data/

# 2. 数据探索（生成图表至 outputs/figures/）
python src/explore_data.py --train data/train.csv --val data/val.csv --label_map data/label_map.json

# 3. 训练（默认 cls 池化，可切换）
python src/train.py --train data/train.csv --val data/val.csv --label_map data/label_map.json --pool cls --epochs 3 --batch_size 16 --max_length 512

# 4. 评估
python src/evaluate.py --val data/val.csv --label_map data/label_map.json --checkpoint outputs/checkpoints/best_cls.pt --pool cls

# 5. 预测（生成 outputs/predictions.csv）
python src/predict.py --predict data/input.csv --checkpoint outputs/checkpoints/best_cls.pt --pool cls --output outputs/predictions.csv
```

---

## 实验管理

### 切换池化策略
```bash
python src/train.py --pool mean   # 或 max
```
不同策略的 checkpoint 会保存为 `outputs/checkpoints/best_{pool}.pt`。

### 启用类别加权
若数据类别严重不均衡，可添加参数：
```bash
python src/train.py --use_class_weight
```

### 调整序列长度
根据数据探索结果调整 `--max_length`（拼接文本通常较长，建议 512）：
```bash
python src/train.py --max_length 256
```

### 自定义训练
```bash
python src/train.py --batch_size 32 --lr_bert 2e-5 --lr_cls 1e-4 --warmup_ratio 0.1 --epochs 5
```

---

## LLM 方案（可选）

### Zero‑Shot 分类（Qwen2‑0.5B）
```bash
# 需下载模型（首次运行自动缓存）
python src_llm/classify_llm.py --val data/val.csv --label_map data/label_map.json --output outputs/llm_zero_shot_results.json
```

### SFT 指令微调（LoRA）
```bash
# 训练
python src_llm/train_sft.py --train data/train.csv --val data/val.csv --label_map data/label_map.json --output_dir outputs/sft_adapter/ --epochs 3 --batch_size 2

# 预测（输出追加到 outputs/predictions.csv）
python src_llm/evaluate_sft.py --adapter outputs/sft_adapter/ --predict data/input.csv --label_map data/label_map.json --output outputs/predictions.csv
```

---

## 结果解读

- **`outputs/predictions.csv`**：原始 `input.csv` 增加一列 `predicted_label`，所有 G 行均被填充预测结果。
- **训练日志**：`outputs/train_log_{pool}.json` 记录每 epoch 损失和准确率。
- **混淆矩阵**：运行 `evaluate.py` 后保存至 `outputs/figures/confusion_matrix.png`。

---

## 常见问题

| 问题 | 解决方法 |
|------|--------|
| `CUDA out of memory` | 减小 `--batch_size`，例如 4 或 8 |
| 训练时间过长 | 切换 `--pool cls`，或减少 `--epochs` |
| `label_map.json` 编码错误 | 确保 `input.csv` 用 UTF-8 保存，或在 `load_data.py` 中指定编码 |
| 拼接后文本超过 512 token | 检查 `explore_data.py` 输出的长度分布，调整 `--max_length` 或过滤掉极长样本 |
| LLM 输出无法解析 | 查看 `outputs/llm_zero_shot_results.json`，手动分析错误样例，调整 prompt |