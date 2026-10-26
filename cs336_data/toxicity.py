from __future__ import annotations
from pathlib import Path
import fasttext
from cs336_data.common import get_shared_assets_path

# 模型单例缓存
_NSFW_MODEL = None
_TOXIC_MODEL = None


def get_nsfw_model():
    global _NSFW_MODEL
    if _NSFW_MODEL is None:
        model_path = get_shared_assets_path() / "classifiers" / "dolma_fasttext_nsfw_jigsaw_model.bin"
        if not model_path.exists():
            model_path = Path("local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
        if not model_path.exists():
            raise FileNotFoundError(f"NSFW model not found at {model_path}")
        _NSFW_MODEL = fasttext.load_model(str(model_path))
    return _NSFW_MODEL


def get_toxic_model():
    global _TOXIC_MODEL
    if _TOXIC_MODEL is None:
        model_path = get_shared_assets_path() / "classifiers" / "dolma_fasttext_hatespeech_jigsaw_model.bin"
        if not model_path.exists():
            model_path = Path("local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
        if not model_path.exists():
            raise FileNotFoundError(f"Toxic model not found at {model_path}")
        _TOXIC_MODEL = fasttext.load_model(str(model_path))
    return _TOXIC_MODEL


def classify_nsfw(text: str) -> tuple[str, float]:
    """
    预测文本是否为 NSFW。
    返回: ('nsfw' 或 'non-nsfw', 置信度)
    """
    if not text or not text.strip():
        return ("non-nsfw", 1.0)

    model = get_nsfw_model()
    clean_text = text.replace("\n", " ").strip()
    labels, probs = model.predict(clean_text, k=1)
    if not labels:
        return ("non-nsfw", 1.0)

    raw_label = labels[0].replace("__label__", "")
    score = float(probs[0])

    # 规范化对齐测试要求：'nsfw' vs 'non-nsfw'
    if "non" in raw_label or "safe" in raw_label or raw_label == "0":
        return ("non-nsfw", score)
    else:
        return ("nsfw", score)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """
    预测文本是否属于有毒言论。
    返回: ('toxic' 或 'non-toxic', 置信度)
    """
    if not text or not text.strip():
        return ("non-toxic", 1.0)

    model = get_toxic_model()
    clean_text = text.replace("\n", " ").strip()
    labels, probs = model.predict(clean_text, k=1)
    if not labels:
        return ("non-toxic", 1.0)

    raw_label = labels[0].replace("__label__", "")
    score = float(probs[0])

    # 规范化对齐测试要求：'toxic' vs 'non-toxic'
    if "non" in raw_label or "clean" in raw_label or raw_label == "0":
        return ("non-toxic", score)
    else:
        return ("toxic", score)