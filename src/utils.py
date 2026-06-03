import os
import subprocess
import sys


HF_MIRROR = "https://hf-mirror.com"
MODELSCOPE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")


def ensure_bert_chinese(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bert-base-chinese")

    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        required_config = os.path.exists(os.path.join(model_dir, "config.json"))
        required_model = os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) or os.path.exists(os.path.join(model_dir, "model.safetensors"))
        required_tokenizer = os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
        if required_config and required_model and required_tokenizer:
            print(f"[utils] 使用本地模型: {model_dir}")
            return model_dir

    ms_dirs = [
        os.path.join(MODELSCOPE_CACHE, "models", "google-bert", "bert-base-chinese"),
        os.path.join(MODELSCOPE_CACHE, "google-bert", "bert-base-chinese"),
        os.path.join(MODELSCOPE_CACHE, "iic", "nlp_bert-base-chinese"),
    ]
    for ms_dir in ms_dirs:
        if os.path.exists(ms_dir):
            print(f"[utils] 使用 ModelScope 缓存: {ms_dir}")
            return ms_dir

    try:
        print("[utils] 尝试从 ModelScope 下载 bert-base-chinese...")
        subprocess.run(
            [sys.executable, "-c",
             "from modelscope import snapshot_download; snapshot_download('iic/nlp_bert-base-chinese', cache_dir=None)"],
            check=True, capture_output=False
        )
        for ms_dir in ms_dirs:
            if os.path.exists(ms_dir):
                return ms_dir
    except Exception as e:
        print(f"[utils] ModelScope 下载失败: {e}")

    os.makedirs(model_dir, exist_ok=True)
    print(f"[utils] 尝试从 HF 镜像下载到本地: {model_dir}")
    for key in ["HF_ENDPOINT", "HF_HUB_ENDPOINT"]:
        os.environ[key] = HF_MIRROR

    try:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        model = BertModel.from_pretrained("bert-base-chinese")
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        print(f"[utils] 模型已保存到: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"[utils] 所有下载方式均失败: {e}")
        raise


def safe_load_bert_tokenizer(tokenizer_name="bert-base-chinese"):
    local_path = ensure_bert_chinese()
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(local_path)


def safe_load_bert_model(model_name="bert-base-chinese"):
    local_path = ensure_bert_chinese()
    from transformers import BertModel
    return BertModel.from_pretrained(local_path)