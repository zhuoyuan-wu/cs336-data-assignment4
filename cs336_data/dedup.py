# cs336_data/dedup.py
from __future__ import annotations

import os
from pathlib import Path
import re
import string
import unicodedata
from collections import Counter, defaultdict

import mmh3
from xopen import xopen


# =====================================================================
# 3.1 Exact Line Deduplication (精确行去重)
# =====================================================================

def hash_line(line: str) -> int:
    """使用 64 位 MurmurHash3 将行映射为整型，大幅降低全局哈希计数器的内存占用"""
    return mmh3.hash64(line.encode("utf-8"))[0]


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    两遍扫描精确行去重：
    Pass 1: 遍历所有输入文件，统计全局每行哈希的频次。
    Pass 2: 重新遍历各文件，仅保留全局频次等于 1 的行，原样写到 output_directory。
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    line_counts = Counter()

    # Pass 1: 统计全局词频
    for filepath in input_files:
        with xopen(filepath, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_counts[hash_line(line)] += 1

    # Pass 2: 过滤重写
    for filepath in input_files:
        path_obj = Path(filepath)
        out_path = output_dir / path_obj.name

        with xopen(filepath, "rt", encoding="utf-8", errors="ignore") as f_in, \
             xopen(out_path, "wt", encoding="utf-8") as f_out:
            for line in f_in:
                if line_counts[hash_line(line)] == 1:
                    f_out.write(line)


# =====================================================================
# 3.2 MinHash + LSH Document Deduplication (模糊文档去重)
# =====================================================================

def normalize_text(text: str) -> str:
    """
    按照 RefinedWeb 标准对文档进行归一化：
    1. NFD Unicode 分解并移除重音修饰符 (Mn)
    2. 全小写化
    3. 移除所有标点符号
    4. 压缩多余连续空格和换行为单个空格
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    """提取词级 N-gram 集合"""
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_minhash_signature(ngrams: set[tuple[str, ...]], num_hashes: int) -> list[int]:
    """使用具有不同 seed 的 MurmurHash3 生成长度为 num_hashes 的 MinHash 签名"""
    if not ngrams:
        return [0] * num_hashes

    # 预编码为字节串以加速哈希计算
    ngram_bytes_list = [" ".join(ng).encode("utf-8") for ng in ngrams]

    signature = []
    for seed in range(num_hashes):
        min_val = min(
            mmh3.hash(ng_bytes, seed=seed, signed=False) for ng_bytes in ngram_bytes_list
        )
        signature.append(min_val)
    return signature


def jaccard_similarity(s1: set, s2: set) -> float:
    """计算两集合的真实 Jaccard 相似度"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union


class UnionFind:
    """并查集数据结构，用于无向重复图的连通分量聚类"""
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    MinHash + LSH 文档去重算法全流程：
    1. 文本标准化与 N-gram 提取
    2. 计算每篇文档的 MinHash 签名
    3. LSH 分 Band 分桶匹配候选对
    4. 精确 Jaccard 校验
    5. 并查集连通图聚类，每个簇保留一篇代表文档，剔除其余重复副本
    6. 将保留的文档原样写入目标目录
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_band = num_hashes // num_bands
    assert num_hashes == num_bands * rows_per_band, "num_hashes 必须能够被 num_bands 整除"

    doc_paths = [Path(p) for p in input_files]
    doc_ngrams = {}
    doc_signatures = {}

    # 1. 提取所有文档的 N-gram 与签名
    for p in doc_paths:
        with xopen(p, "rt", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        norm_text = normalize_text(raw_text)
        words = norm_text.split()
        ng_set = get_ngrams(words, ngrams)
        doc_ngrams[p] = ng_set
        doc_signatures[p] = compute_minhash_signature(ng_set, num_hashes)

    # 2. LSH 局部敏感分桶找候选对
    candidate_pairs = set()
    for band_idx in range(num_bands):
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band
        buckets = defaultdict(list)

        for p in doc_paths:
            band_key = tuple(doc_signatures[p][start_row:end_row])
            buckets[band_key].append(p)

        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        doc_a, doc_b = bucket_docs[i], bucket_docs[j]
                        if doc_a != doc_b:
                            pair = (min(doc_a, doc_b), max(doc_a, doc_b))
                            candidate_pairs.add(pair)

    # 3. 精确 Jaccard 校验与聚类并查集构建
    uf = UnionFind(doc_paths)
    for doc_a, doc_b in candidate_pairs:
        sim = jaccard_similarity(doc_ngrams[doc_a], doc_ngrams[doc_b])
        if sim >= jaccard_threshold:
            uf.union(doc_a, doc_b)

    # 4. 连通分支聚类，每个簇仅保留代表节点
    clusters = defaultdict(list)
    for p in doc_paths:
        root = uf.find(p)
        clusters[root].append(p)

    docs_to_keep = set()
    for root, cluster_docs in clusters.items():
        # 稳定排序保证确定性，保留第一个
        cluster_docs.sort()
        docs_to_keep.add(cluster_docs[0])

    # 5. 将选中的文档写入 output_directory
    for p in docs_to_keep:
        out_path = output_dir / p.name
        with xopen(p, "rt", encoding="utf-8", errors="ignore") as f_in, \
             xopen(out_path, "wt", encoding="utf-8") as f_out:
            f_out.write(f_in.read())