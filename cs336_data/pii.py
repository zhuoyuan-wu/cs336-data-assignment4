# cs336_data/pii.py
from __future__ import annotations
import re

# 1. 邮箱正则：匹配常规 email 格式
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# 2. 电话号码正则：覆盖 US 常用格式及测试用例：
# 2831823829, (283)-182-3829, (283) 182 3829, 283-182-3829 等
PHONE_PATTERN = re.compile(
    r"(?:(?:\(\d{3}\)|\b\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b)"
)

# 3. IPv4 正则：精确匹配 0-255 的 4 个字段，并带词边界防止误匹配浮点数
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
IPV4_PATTERN = re.compile(
    rf"\b(?:{_OCTET}\.){{3}}{_OCTET}\b"
)


def mask_emails(text: str) -> tuple[str, int]:
    """
    替换所有电子邮箱地址为 |||EMAIL_ADDRESS|||，并返回脱敏后的文本与替换次数。
    """
    masked_text, count = EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||", text)
    return masked_text, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    替换所有电话号码为 |||PHONE_NUMBER|||，并返回脱敏后的文本与替换次数。
    """
    masked_text, count = PHONE_PATTERN.subn("|||PHONE_NUMBER|||", text)
    return masked_text, count


def mask_ips(text: str) -> tuple[str, int]:
    """
    替换所有 IPv4 地址为 |||IP_ADDRESS|||，并返回脱敏后的文本与替换次数。
    """
    masked_text, count = IPV4_PATTERN.subn("|||IP_ADDRESS|||", text)
    return masked_text, count