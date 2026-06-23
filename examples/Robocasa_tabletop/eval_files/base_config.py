from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

try:
    from pydantic import BaseModel
except ModuleNotFoundError:
    BaseModel = None


if BaseModel is not None:

    class ModalityConfig(BaseModel):
        """Configuration for a modality."""

        delta_indices: list[int]
        """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
        modality_keys: list[str]
        """The keys to load for the modality in the dataset."""

else:

    @dataclass
    class ModalityConfig:
        """Configuration for a modality."""

        delta_indices: list[int]
        modality_keys: list[str]


class BasePolicy(ABC):
    """Base interface for evaluation policies."""

    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError
