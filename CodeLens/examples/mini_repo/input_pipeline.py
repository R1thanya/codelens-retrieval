"""Small synthetic example used to demonstrate retrieval ranking."""


def normalize_input(raw_text: str) -> str:
    """Trim user input before passing it into the application."""
    cleaned = raw_text.strip()
    return dispatch(cleaned)


def dispatch(value: str) -> str:
    return value


def main(raw_text: str) -> str:
    return normalize_input(raw_text)
