# scripts/filter_dataset.py
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import os
from pathlib import Path
from tqdm import tqdm
from xopen import xopen
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.langid import identify_language
from cs336_data.quality import gopher_quality_filter, classify_quality
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech
from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips


def process_single_wet(wet_path: Path, output_file: Path) -> Counter:
    """
    处理单个 .warc.wet.gz 文件，输出通过所有过滤器的文本，
    并返回该分片的统计计数器。
    """
    stats = Counter()

    with xopen(wet_path, "rb") as stream, \
         xopen(output_file, "wt", encoding="utf-8") as out:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
            stats["total_records"] += 1
            payload = record.reader.read()
            text = payload.decode("utf-8", errors="replace").strip()

            if not text:
                stats["discard_empty"] += 1
                continue

            # 1. 语言识别 (English >= 0.7)
            lang, lang_score = identify_language(text)
            if lang != "en" or lang_score < 0.70:
                stats["discard_language"] += 1
                continue

            # 2. Gopher 启发式规则过滤
            if not gopher_quality_filter(text):
                stats["discard_gopher"] += 1
                continue

            # 3. 有害内容过滤 (NSFW & Toxic)
            nsfw_label, _ = classify_nsfw(text)
            if nsfw_label == "nsfw":
                stats["discard_nsfw"] += 1
                continue

            toxic_label, _ = classify_toxic_speech(text)
            if toxic_label == "toxic":
                stats["discard_toxic"] += 1
                continue

            # 4. 质量分类器过滤 (保留高质量)
            quality_label, _ = classify_quality(text)
            if quality_label == "cc":
                # 过滤低质 CC 垃圾
                stats["discard_quality"] += 1
                continue

            # 5. PII 脱敏
            text, _ = mask_emails(text)
            text, _ = mask_phone_numbers(text)
            text, _ = mask_ips(text)

            stats["kept_records"] += 1
            # 写入文档，文档间以双换行或特定分隔符隔开
            out.write(text + "\n\n")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="local-shared-data/CC")
    parser.add_argument("--output-dir", type=str, default="local-shared-data/filtered")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(list(input_dir.glob("*.wet.gz")) + list(input_dir.glob("*.warc.wet.gz")))
    if not wet_files:
        print(f"No WET files found in {input_dir}")
        return

    print(f"Found {len(wet_files)} WET files. Launching {args.workers} workers...")

    total_stats = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_wet,
                wet_file,
                output_dir / f"clean_{wet_file.name.replace('.gz', '.txt')}"
            ): wet_file
            for wet_file in wet_files
        }

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Filtering WET"):
            sub_stats = fut.result()
            total_stats.update(sub_stats)

    print("\n" + "=" * 50)
    print("DATA FILTERING BREAKDOWN REPORT:")
    print("=" * 50)
    total = total_stats["total_records"]
    kept = total_stats["kept_records"]
    discarded = total - kept
    print(f"Total Records Processed: {total:,}")
    print(f"Records Kept:           {kept:,} ({kept / total * 100:.2f}%)")
    print(f"Total Discarded:        {discarded:,} ({discarded / total * 100:.2f}%)")
    print("-" * 50)
    for k, v in total_stats.items():
        if k.startswith("discard_"):
            pct_of_discarded = (v / discarded * 100) if discarded else 0
            print(f"  {k:20s}: {v:,} ({pct_of_discarded:.2f}% of discarded)")
    print("=" * 50)


if __name__ == "__main__":
    main()