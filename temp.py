import json
import re
import unicodedata
from jiwer import wer

HEX_TOKEN_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")

def fix_utf8_bytes(text: str) -> str:
    """
    Convert sequences like <0xE0><0xB2><0xA2> into proper UTF-8 characters.
    Invalid byte sequences are dropped safely.
    """
    tokens = HEX_TOKEN_RE.split(text)
    output = []
    byte_buffer = []

    for i, part in enumerate(tokens):
        if i % 2 == 1:
            # hex byte
            byte_buffer.append(int(part, 16))
        else:
            # flush any buffered bytes
            if byte_buffer:
                try:
                    output.append(bytes(byte_buffer).decode("utf-8"))
                except UnicodeDecodeError:
                    # drop invalid bytes
                    pass
                byte_buffer = []
            output.append(part)

    # flush remaining bytes
    if byte_buffer:
        try:
            output.append(bytes(byte_buffer).decode("utf-8"))
        except UnicodeDecodeError:
            pass

    return "".join(output)


def normalize_text(text: str) -> str:
    """
    Unicode-safe normalization for ASR evaluation.
    """
    text = fix_utf8_bytes(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()


def compute_wer_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refs = []
    hyps = []

    for item in data:
        ref = normalize_text(item["ground_truth"])
        hyp = normalize_text(item["prediction"])
        refs.append(ref)
        hyps.append(hyp)

    return wer(refs, hyps)


if __name__ == "__main__":
    path = "/Users/chaitanyakartik/Projects/asr-finetuning/docs/console_logs/benchmark_incomplete_v3.json"  # change this
    score = compute_wer_from_json(path)
    print(f"Normalized WER: {score:.4f}")
