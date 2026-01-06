class UserFacingError(Exception):
    """Error safe to display in the UI."""


def friendly_error(message: str) -> str:
    return f"Error: {message}"
