from __future__ import annotations


class RegistryError(ValueError):
    """Base class for expected model registry failures."""


class RegistryNotFoundError(RegistryError):
    """A referenced registry record does not exist."""


class RegistryConflictError(RegistryError):
    """A registry operation conflicts with existing state."""


class RegistryValidationError(RegistryError):
    """A registry input or state transition is invalid."""
