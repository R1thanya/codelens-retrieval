"""Unrelated example: a locale-prefix check."""


def check_locale(value: str) -> bool:
    prefix = value[:5]
    return prefix == "en-US"
