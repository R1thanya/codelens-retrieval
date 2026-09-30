def parse_bearer_token(value):
    """Extract the credential that follows the Bearer scheme."""
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def authorize_request(headers, token_store, now):
    """Validate token and expiry before allowing a request."""
    value = headers.get("Authorization", "")
    token = parse_bearer_token(value)
    claims = token_store.get_claims(token)
    if claims is None or claims["expires_at"] <= now:
        return False
    return token_store.is_valid(token)
