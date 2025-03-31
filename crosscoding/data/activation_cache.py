import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformer_lens import HookedTransformer  # type: ignore

from crosscoding.log import logger


class ActivationsCache:
    """Handles caching and loading of model activations to/from disk."""

    def __init__(self, cache_dir: str, use_mmap: bool = False):
        """
        Initialize the activations cache.

        Args:
            cache_dir: Directory to store cached activations. If None, caching is disabled.
            use_mmap: Whether to use memory-mapped files for loading large cached activations.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_mmap = use_mmap

    def get_cache_key(self, model: HookedTransformer, sequence_BS: torch.Tensor, hookpoints: list[str]) -> str:
        """Generate a unique cache key based on model identifier and input hash."""
        # Access the model key buffer that was registered in build_llms
        if not hasattr(model, "crosscode_model_key"):
            raise ValueError(
                "Model does not have a crosscode_model_key buffer. "
                "Please use the build_llms function to create the model."
            )

        # Convert tensor of ASCII values back to string
        model_key_tensor = cast(torch.Tensor, model.crosscode_model_key)
        model_key = "".join(chr(i) for i in model_key_tensor.tolist())

        # Create a deterministic hash of the input tokens
        sequence_bytes = sequence_BS.cpu().numpy().tobytes()
        input_hash = hashlib.md5(sequence_bytes).hexdigest()
        hookpoints_hash = hashlib.md5("".join(hookpoints).encode()).hexdigest()

        return f"{model_key}_{input_hash}_{hookpoints_hash}"

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cached activation."""
        if not self.cache_dir:
            raise ValueError("Cache directory not set")
        return self.cache_dir / f"{cache_key}.npy"

    def load_activations(self, cache_key: str, device: torch.device) -> torch.Tensor | None:
        """
        Load cached activations from disk.

        Args:
            cache_key: The unique key for the cached activations
            device: Device to load the tensor to

        Returns:
            The loaded activations tensor, or None if loading fails
        """
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            np_array = np.load(cache_path, mmap_mode="r" if self.use_mmap else None)

            return torch.from_numpy(np_array).to(device)
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None

    def save_activations(self, cache_key: str, activations: torch.Tensor) -> bool:
        """
        Save activations to disk cache.

        Args:
            cache_key: The unique key for the cached activations
            activations: The tensor to cache

        Returns:
            True if saving was successful, False otherwise
        """
        cache_path = self.get_cache_path(cache_key)

        # Ensure the parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save as numpy array for better compatibility with mmap
            np_array = activations.cpu().numpy()
            np.save(cache_path, np_array)
            return True
        except Exception as e:
            logger.warning(f"Failed to write cache to {cache_path}: {e}")
            raise e
            # return False
