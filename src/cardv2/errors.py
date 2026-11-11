class CardError(Exception):
    """A failure the caller can act on. `code` is stable and machine-readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
