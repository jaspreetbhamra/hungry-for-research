import re
import unicodedata


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_header_footer(text: str) -> str:
    # Hook: implement project-specific header/footer stripping if needed.
    # For now, just return normalized.
    return normalize_text(text)
