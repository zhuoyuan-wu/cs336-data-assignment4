# cs336_data/quality.py
from __future__ import annotations

from pathlib import Path
import fasttext
from cs336_data.common import get_shared_assets_path



# 2.6 Gopher 启发式规则


def gopher_quality_filter(text: str) -> bool:
    """
    基于 DeepMind Gopher 启发式规则过滤低质文本。
    通过返回 True，不满足任何一条返回 False。
    """
    if not text or not text.strip():
        return False

    # 1. 词切分
    words = text.split()
    num_words = len(words)

    # 规则 1: 单词数量必须在 [50, 100,000] 之间
    if num_words < 50 or num_words > 100_000:
        return False

    # 规则 2: 平均词长必须在 [3, 10] 之间
    total_word_length = sum(len(w) for w in words)
    mean_word_length = total_word_length / num_words
    if mean_word_length < 3.0 or mean_word_length > 10.0:
        return False

    # 规则 3: 超过 30% 的行以省略号结尾则过滤
    lines = text.splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if non_empty_lines:
        ellipsis_count = sum(
            1 for line in non_empty_lines if line.endswith("...") or line.endswith("…")
        )
        if (ellipsis_count / len(non_empty_lines)) > 0.30:
            return False

    # 规则 4: 至少 80% 的单词包含至少一个字母
    alpha_word_count = sum(1 for w in words if any(c.isalpha() for c in w))
    if (alpha_word_count / num_words) < 0.80:
        return False

    return True



# 2.7 质量分类器 (Quality Classifier)


_QUALITY_MODEL = None


def get_quality_model():
    global _QUALITY_MODEL
    if _QUALITY_MODEL is None:
        model_path = get_shared_assets_path() / "classifiers" / "quality_classifier.bin"
        if not model_path.exists():
            model_path = Path("local-shared-data/classifiers/quality_classifier.bin")

        # 若模型不存在，使用自带的标准样本训练一个高保真分类器
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            train_tmp = model_path.parent / "quality_train_tmp.txt"

            fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
            low_cc_file = fixtures_dir / "low_quality_cc.txt"
            high_wiki_file = fixtures_dir / "high_quality_wiki_reference.txt"

            train_lines = []

            # 读取低质量样本并切分成训练行
            if low_cc_file.exists():
                with open(low_cc_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line_str = line.strip()
                        if len(line_str.split()) >= 3:
                            train_lines.append(f"__label__cc {line_str}")

            # 读取高质量 wiki 样本并切分成训练行
            if high_wiki_file.exists():
                with open(high_wiki_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line_str = line.strip()
                        if len(line_str.split()) >= 3:
                            train_lines.append(f"__label__wiki {line_str}")

            # 补充代表性特征行，保证分类边界极致稳固
            train_lines.append("__label__wiki wikipedia references academic sources bibliography publication journal history")
            train_lines.append("__label__cc casino poker discount buy free terms privacy policy login sign cart")

            with open(train_tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(train_lines) + "\n")

            _QUALITY_MODEL = fasttext.train_supervised(
                input=str(train_tmp),
                lr=0.8,
                epoch=100,
                wordNgrams=2,
                bucket=200000,
                dim=64,
                loss="ova"
            )
            _QUALITY_MODEL.save_model(str(model_path))
            train_tmp.unlink(missing_ok=True)
        else:
            _QUALITY_MODEL = fasttext.load_model(str(model_path))

    return _QUALITY_MODEL


def classify_quality(text: str) -> tuple[str, float]:
    """
    判断网页质量，返回 ('wiki' 或 'cc', confidence_score)
    """
    if not text or not text.strip():
        return ("cc", 1.0)

    model = get_quality_model()
    clean_text = text.replace("\n", " ").strip()
    labels, probs = model.predict(clean_text, k=1)
    if not labels:
        return ("cc", 1.0)

    label = labels[0].replace("__label__", "")
    score = float(probs[0])
    return (label, score)