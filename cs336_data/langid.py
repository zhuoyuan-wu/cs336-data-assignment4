# cs336_data/langid.py
from __future__ import annotations
from pathlib import Path
import fasttext
from cs336_data.common import get_shared_assets_path

# 全局单例模型缓存
_LANGID_MODEL = None


def get_langid_model():
    global _LANGID_MODEL
    if _LANGID_MODEL is None:
        model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
        if not model_path.exists():
            # 兼容从项目根目录调用的相对路径
            model_path = Path("local-shared-data/classifiers/lid.176.bin")
        if not model_path.exists():
            raise FileNotFoundError(
                f"FastText language identification model not found at: {model_path}\n"
                "Please download it via wget or scripts/download_data.py"
            )
        # fasttext load_model
        _LANGID_MODEL = fasttext.load_model(str(model_path))
    return _LANGID_MODEL


def identify_language(text: str) -> tuple[str, float]:
    """
    识别 Unicode 文本的主语言。
    返回: (language_code, confidence_score)
    例如: ('en', 0.985), ('zh', 0.999)
    """
    if not text or not text.strip():
        return ("unknown", 0.0)

    model = get_langid_model()

    # 1. 预处理：替换换行符为空格，避免 fastText C++ 引擎报错或截断
    cleaned_text = text.replace("\n", " ").strip()
    if not cleaned_text:
        return ("unknown", 0.0)

    # 2. 预测 Top-1 结果
    labels, probs = model.predict(cleaned_text, k=1)
    if not labels:
        return ("unknown", 0.0)

    # 3. 标签剥离前缀：'__label__en' -> 'en'
    raw_label = labels[0]
    lang = raw_label.replace("__label__", "")
    score = float(probs[0])

    return (lang, score)