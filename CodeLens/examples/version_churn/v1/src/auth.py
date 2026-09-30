def parse_bearer_token(value):
    """Extract the credential that follows the Bearer scheme."""
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def authorize_request(headers, token_store):
    """Validate a bearer token on incoming requests."""
    value = headers.get("Authorization", "")
    token = parse_bearer_token(value)
    return token_store.is_valid(token)
