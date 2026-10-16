class ConfigurationError(RuntimeError):
    """Required provider configuration is missing."""


class ProviderError(RuntimeError):
    """An upstream model or embedding operation failed."""


class InvalidDocument(ValueError):
    """An uploaded document cannot be ingested."""
