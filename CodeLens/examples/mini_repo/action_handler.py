"""Unrelated example: dispatch an action without normalizing input."""


def perform_action(actor: str, action: str) -> str:
    if actor and action:
        return f"{actor}:{action}"
    return "noop"
