"""Registry for OSA model backends."""

from __future__ import annotations

from osa.models.interface import ModelInterface


class ModelRegistry:
    """Store and retrieve registered OSA models."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInterface] = {}
        self._default_model: str | None = None

    def register(
        self,
        model: ModelInterface,
        *,
        as_default: bool = False,
    ) -> None:
        """Register a model backend."""
        name = model.model_name

        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered.")

        self._models[name] = model

        if as_default:
            self._default_model = name

    def get(self, name: str) -> ModelInterface:
        """Return a registered model by name."""
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Model '{name}' is not registered.") from exc

    def get_default(self) -> ModelInterface:
        """Return the configured default model."""
        if self._default_model is None:
            raise RuntimeError("No default model has been configured.")

        return self.get(self._default_model)

    def names(self) -> tuple[str, ...]:
        """Return registered model names."""
        return tuple(self._models)

    def __len__(self) -> int:
        """Return the number of registered models."""
        return len(self._models)
