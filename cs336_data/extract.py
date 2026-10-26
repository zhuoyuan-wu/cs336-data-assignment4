from __future__ import annotations
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    if not html_bytes:
        return ""

    # 1. 优先尝试 UTF-8 快速解码
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 2. UTF-8 失败时，使用 resiliparse 的底层编码探测
        encoding = detect_encoding(html_bytes)
        try:
            html_str = html_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError, TypeError):
            # 3. 兜底容错
            html_str = html_bytes.decode("utf-8", errors="replace")

    # 4. 调用 extract_plain_text 提取纯文本
    # 默认参数保留常规换行与段落结构，不强制开启 main_content 启发式过滤
    return extract_plain_text(html_str)