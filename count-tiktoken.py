from collections import Counter
import tiktoken

HAN_RANGES = (
 (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
 (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
 (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x30000, 0x3134F),
 (0x31350, 0x323AF),
)

def is_han(ch: str) -> bool:
 cp = ord(ch)
 return any(a <= cp <= b for a, b in HAN_RANGES)

def pure_han_len(text: str):
 if not text:
     return None
 return len(text) if all(is_han(ch) for ch in text) else None

enc = tiktoken.get_encoding("cl100k_base")
n_vocab = enc.n_vocab
special_ids = set(enc._special_tokens.values())

pure = 0
counter = Counter()
checked = 0

for idx in range(n_vocab):
 if idx in special_ids:
     continue
 checked += 1
 try:
     s = enc.decode_single_token_bytes(idx).decode("utf-8")
 except Exception:
     continue
 L = pure_han_len(s)
 if L is not None:
     pure += 1
     counter[L] += 1

print("model: tiktoken/cl100k_base")
print(f"vocab_size: {n_vocab}")
print(f"special_tokens_skipped: {len(special_ids)}")
print(f"checked_tokens: {checked}")
print(f"pure_han_tokens: {pure}")
print(f"ratio: {pure/checked*100:.4f}%")
print("Distribution by Han char count:")
for k in sorted(counter):
 c = counter[k]
 print(f"{k:>2} chars: {c:<6} ({c/pure*100:.2f}% of pure Han tokens)")
