#!/usr/bin/env python3
"""
Count how many tokens in a Qwen tokenizer decode to pure Han characters.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Iterable

from transformers import AutoTokenizer


# Common CJK/Han blocks (including extensions and compatibility ideographs).
HAN_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E/F
    (0x2CEB0, 0x2EBEF),  # Extension F/I (newer allocations included)
    (0x30000, 0x3134F),  # Extension G
    (0x31350, 0x323AF),  # Extension H
)


def is_han_char(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in HAN_RANGES)


def is_pure_han(text: str) -> bool:
    return bool(text) and all(is_han_char(ch) for ch in text)


def pure_han_length(text: str) -> int | None:
    if not text:
        return None
    if not is_pure_han(text):
        return None
    return len(text)


def iter_vocab_ids(vocab: dict[str, int]) -> Iterable[int]:
    return iter(sorted(vocab.values()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 Qwen tokenizer 中“解码后是纯汉字”的词元数量。"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face 模型名或本地 tokenizer 路径",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="传给 AutoTokenizer.from_pretrained 的 trust_remote_code=True",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=30,
        help="展示多少个命中的示例（默认: 30）",
    )
    parser.add_argument(
        "--show-class-samples",
        type=int,
        default=3,
        help="每个“汉字数量类别”展示多少个示例（默认: 3）",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    vocab = tokenizer.get_vocab()
    id_to_raw_token = {idx: token for token, idx in vocab.items()}
    special_ids = set(tokenizer.all_special_ids or [])

    pure_han_ids: list[int] = []
    pure_han_len_counter: Counter[int] = Counter()
    pure_han_ids_by_len: dict[int, list[int]] = defaultdict(list)

    for idx in iter_vocab_ids(vocab):
        if idx in special_ids:
            continue
        decoded = tokenizer.decode(
            [idx], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        han_len = pure_han_length(decoded)
        if han_len is not None:
            pure_han_ids.append(idx)
            pure_han_len_counter[han_len] += 1
            if len(pure_han_ids_by_len[han_len]) < args.show_class_samples:
                pure_han_ids_by_len[han_len].append(idx)

    checked = len(vocab) - len(special_ids)
    ratio = (len(pure_han_ids) / checked * 100) if checked else 0.0

    print(f"model: {args.model}")
    print(f"vocab_size: {len(vocab)}")
    print(f"special_tokens_skipped: {len(special_ids)}")
    print(f"checked_tokens: {checked}")
    print(f"pure_han_tokens: {len(pure_han_ids)}")
    print(f"ratio: {ratio:.4f}%")

    if pure_han_ids:
        print("\nDistribution by Han char count:")
        for han_len in sorted(pure_han_len_counter):
            count = pure_han_len_counter[han_len]
            pct = count / len(pure_han_ids) * 100
            print(f"{han_len:>2} chars: {count:<6} ({pct:.2f}% of pure Han tokens)")

    if args.show_samples > 0 and pure_han_ids:
        print("\nSamples:")
        for idx in pure_han_ids[: args.show_samples]:
            raw = repr(id_to_raw_token.get(idx, ""))
            decoded = tokenizer.decode(
                [idx], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            print(f"id={idx:<6} raw={raw:<24} decoded={decoded}")

    if args.show_class_samples > 0 and pure_han_ids:
        print("\nSamples by Han char count:")
        for han_len in sorted(pure_han_len_counter):
            print(f"[{han_len} chars]")
            for idx in pure_han_ids_by_len[han_len]:
                raw = repr(id_to_raw_token.get(idx, ""))
                decoded = tokenizer.decode(
                    [idx], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                print(f"  id={idx:<6} raw={raw:<24} decoded={decoded}")


if __name__ == "__main__":
    main()
