import re


def clean_text(text: str) -> str:
    """Remove artifacts and normalize whitespace from transcript text."""
    # Remove music/sound notations like [Music], [Applause]
    text = re.sub(r"\[.*?\]", "", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()
