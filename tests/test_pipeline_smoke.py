# tests/test_pipeline_smoke.py
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

from cs336_data.langid import identify_language
from cs336_data.quality import gopher_quality_filter, classify_quality
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech
from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips




def test_clean_pipeline_smoke(tmp_path):
    # 标准的英语自然语言段落，夹带少量 PII
    natural_text = (
        "Natural language processing has seen rapid advances in recent years with the development "
        "of large scale transformer models. These models demonstrate impressive capabilities across "
        "a wide variety of downstream tasks. "
    ) * 5

    pii_part = (
        "For correspondence, please reach out via email at researcher@stanford.edu or call (650) 723-2300. "
        "The cluster is hosted at 192.168.1.1."
    )
    
    raw_sample = natural_text + " " + pii_part

    # 1. 语言识别验证
    lang, score = identify_language(raw_sample)
    assert lang == "en", f"Expected 'en', got {lang}"
    assert score >= 0.70, f"Score was {score}"

    # 2. Gopher 质量过滤验证
    assert gopher_quality_filter(raw_sample) is True

    # 3. 毒性与 NSFW 验证
    nsfw_label, _ = classify_nsfw(raw_sample)
    assert nsfw_label == "non-nsfw"

    toxic_label, _ = classify_toxic_speech(raw_sample)
    assert toxic_label == "non-toxic"

    # 4. 质量分类器验证
    quality_label, _ = classify_quality(raw_sample)
    assert quality_label in ["wiki", "cc"]

    # 5. PII 脱敏验证
    clean_sample, c1 = mask_emails(raw_sample)
    clean_sample, c2 = mask_phone_numbers(clean_sample)
    clean_sample, c3 = mask_ips(clean_sample)
    assert c1 > 0 and c2 > 0 and c3 > 0
    assert "|||EMAIL_ADDRESS|||" in clean_sample
    assert "|||PHONE_NUMBER|||" in clean_sample
    assert "|||IP_ADDRESS|||" in clean_sample

    # 6. 分词与二进制序列化冒烟
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(clean_sample) + [tokenizer.eos_token_id]
    
    bin_path = tmp_path / "smoke_tokens.bin"
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(bin_path)

    # 7. 二进制反序列化校验
    loaded_arr = np.fromfile(bin_path, dtype=np.uint16)
    assert len(loaded_arr) == len(tokens)
    assert loaded_arr[-1] == tokenizer.eos_token_id