import argparse
import json
import os
import re
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split


DATE_PATTERNS = [
    (re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"), lambda m: m.group().replace("/", "年", 1).replace("/", "月") + "日"),
    (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), lambda m: m.group().replace("-", "年", 1).replace("-", "月") + "日"),
    (re.compile(r"^\d{8}$"), lambda m: (
        f"{m.group()[:4]}年{int(m.group()[4:6])}月{int(m.group()[6:8])}日"
        if 1 <= int(m.group()[4:6]) <= 12 and 1 <= int(m.group()[6:8]) <= 31
        else m.group()
    )),
]

BOOL_KEYWORDS = {"has", "is", "eng", "enable", "show"}


def detect_encoding(filepath):
    with open(filepath, "rb") as f:
        raw = f.read(100000)
    result = chardet.detect(raw)
    return result.get("encoding", "utf-8")


def is_date_value(val_str):
    for pat, _ in DATE_PATTERNS:
        if pat.match(val_str):
            return True
    return False


def convert_date(val_str):
    for pat, converter in DATE_PATTERNS:
        m = pat.match(val_str)
        if m:
            return converter(m)
    return val_str


def is_bool_column(col_name):
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in BOOL_KEYWORDS)


def format_field_value(col_name, raw_value):
    if pd.isna(raw_value):
        return "未知"
    val_str = str(raw_value).strip()
    if val_str in ("", "\\N", "nan", "NaN", "None"):
        return "未知"

    if is_date_value(val_str):
        return convert_date(val_str)

    if is_bool_column(col_name):
        if val_str.upper() in ("Y", "YES", "1", "TRUE"):
            return "是"
        elif val_str.upper() in ("N", "NO", "0", "FALSE"):
            return "否"

    return val_str


def row_to_text(row, feature_columns):
    parts = []
    for col in feature_columns:
        value = format_field_value(col, row.get(col))
        parts.append(f"{col}: {value}")
    return "，".join(parts)


def load_and_process(input_path, output_dir, val_ratio=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    encoding = detect_encoding(input_path)
    print(f"[load_data] 检测到编码: {encoding}")
    try:
        df = pd.read_csv(input_path, encoding=encoding, on_bad_lines="skip")
    except Exception:
        for enc in ["utf-8-sig", "gbk", "gb18030", "latin-1"]:
            try:
                df = pd.read_csv(input_path, encoding=enc, on_bad_lines="skip")
                break
            except Exception:
                continue

    print(f"[load_data] 总行数: {len(df)}, 列数: {len(df.columns)}")

    label_col = "sample_state"
    if label_col not in df.columns:
        raise ValueError(f"缺少必要列: {label_col}")

    label_counts = df[label_col].value_counts()
    print(f"[load_data] {label_col} 分布:\n{label_counts.to_string()}")

    exclude_cols = {label_col}
    feature_columns = [c for c in df.columns if c not in exclude_cols]

    print(f"[load_data] 特征列数量: {len(feature_columns)}")
    df["text"] = df.apply(lambda row: row_to_text(row, feature_columns), axis=1)

    label_mapping = {"B": 0, "W": 1}
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    labeled_df = df[df[label_col].isin(["B", "W"])].copy()
    labeled_df[label_col] = labeled_df[label_col].map(label_mapping)

    predict_df = df[df[label_col] == "G"].copy()

    train_df, val_df = train_test_split(
        labeled_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=labeled_df[label_col],
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    predict_path = os.path.join(output_dir, "predict.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    predict_df.to_csv(predict_path, index=False, encoding="utf-8-sig")

    print(f"[load_data] 训练集: {len(train_df)} 行 -> {train_path}")
    print(f"[load_data] 验证集: {len(val_df)} 行 -> {val_path}")
    print(f"[load_data] 预测集: {len(predict_df)} 行 -> {predict_path}")
    print(f"[load_data] 标签映射: {os.path.join(output_dir, 'label_map.json')}")

    return train_path, val_path, predict_path


def main():
    parser = argparse.ArgumentParser(description="加载 CSV 数据，拼接字段为自然语言并划分数据集")
    parser.add_argument("--input", type=str, default="input.csv", help="输入 CSV 文件路径")
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    load_and_process(args.input, args.output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()