import random
from typing import Literal

import torch
from transformer_lens import HookedTransformer  # type: ignore

from model_diffing.data.activation_cache import ActivationsCache
from model_diffing.log import logger

# shapes:
# H: (harvesting) batch size
# S: sequence length
# P: hookpoints
# D: model d_model

CacheMode = Literal["no_cache", "cache", "cache_with_mmap"]


class MemoryTracker:
    """Lightweight memory tracker for frequent function calls"""

    def __init__(self, threshold_mb=100, sample_rate=0.01):
        self.threshold_mb = threshold_mb
        self.sample_rate = sample_rate
        self.call_count = 0
        self.last_allocated = 0
        self.record_detailed = False
        self.peak_allocated = 0
        self.total_leaks = 0

    def start_tracking(self):
        self.last_allocated = torch.cuda.memory_allocated()
        if self.record_detailed:
            # Use correct parameters for memory history recording
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="all", max_entries=10000)
        return self.last_allocated

    def stop_tracking(self, start_allocated, operation_name):
        end_allocated = torch.cuda.memory_allocated()
        delta_mb = (end_allocated - start_allocated) / (1024**2)

        # Update statistics
        self.peak_allocated = max(self.peak_allocated, end_allocated)

        # Check for potential memory leak
        if delta_mb > 1.0:  # Consider any residual over 1MB as potential leak
            self.total_leaks += delta_mb

        # Only report issues over threshold or during detailed sampling
        do_detailed_report = abs(delta_mb) >= self.threshold_mb or (
            self.record_detailed and random.random() < self.sample_rate
        )

        if do_detailed_report:
            print(
                f"\n⚠️ Memory {'increased' if delta_mb > 0 else 'decreased'} by {abs(delta_mb):.2f}MB in {operation_name}"
            )

            if self.record_detailed:
                # Get and analyze memory snapshot
                snapshot = torch.cuda.memory._snapshot()
                active_segments = [s for s in snapshot.segments if s.is_active]

                if active_segments:
                    largest = max(active_segments, key=lambda s: s.size)
                    size_mb = largest.size / (1024**2)
                    print(f"Largest active allocation: {size_mb:.2f}MB")

                    if hasattr(largest, "frames") and largest.frames:
                        for frame in largest.frames[-3:]:
                            if not "site-packages/torch/" in frame.filename:
                                print(f"  {frame.filename}:{frame.line}")

                # Turn off recording to reduce overhead
                torch.cuda.memory._record_memory_history(enabled=None)

        return end_allocated, delta_mb


# Create a singleton tracker
memory_tracker = MemoryTracker(threshold_mb=4000)


class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer],
        hookpoints: list[str],
        cache_dir: str | None = None,
        cache_mode: CacheMode = "no_cache",
    ):
        if len({llm.cfg.d_model for llm in llms}) != 1:
            raise ValueError("All models must have the same d_model")
        self._llms = llms
        self._hookpoints = hookpoints

        # Set up the activations cache
        self._cache = None
        if cache_mode != "no_cache":
            if not cache_dir:
                raise ValueError("cache_mode is enabled but no cache_dir provided; caching will be disabled")
            self._cache = ActivationsCache(cache_dir=cache_dir, use_mmap=cache_mode == "cache_with_mmap")

        self.num_models = len(llms)
        self._num_hookpoints = len(hookpoints)
        self._layer_to_stop_at = self._get_layer_to_stop_at()

        self._mem_call_count = 0
        self._mem_total_leaked_MB = 0.
        self._mem_last_warning = 0


    def _get_layer_to_stop_at(self) -> int:
        last_needed_layer = max(_get_layer(name) for name in self._hookpoints)
        layer_to_stop_at = last_needed_layer + 1
        logger.info(f"computed last needed layer: {last_needed_layer}, stopping at {layer_to_stop_at}")
        return layer_to_stop_at

    def _names_filter(self, name: str) -> bool:
        return name in self._hookpoints  # not doing any fancy hash/set usage as this list is tiny

    def _compute_model_activations_HSPD(
        self,
        model: HookedTransformer,
        sequence_HS: torch.Tensor,
    ) -> torch.Tensor:
        """Compute activations by running the model with memory monitoring."""
        # Initialize tracking attributes if they don't exist

        self._mem_call_count += 1

        # Track memory only on sampled calls
        # start_mem_MB = torch.cuda.memory_allocated() / (1024**2)

        # Original function logic
        with torch.no_grad():
            _, cache = model.run_with_cache(
                sequence_HS,
                names_filter=self._names_filter,
                stop_at_layer=self._layer_to_stop_at,
            )

        # Stack the activations (note: no clone() here - may help memory issues)
        activations_HSPD = torch.stack([cache[name] for name in self._hookpoints], dim=2)

        # Free memory immediately
        del cache
        return activations_HSPD

        # Check for memory issues on sampled calls
        end_mem_MB = torch.cuda.memory_allocated() / (1024**2)
        delta_MB = end_mem_MB - start_mem_MB

        # If memory increased significantly, that's potentially a leak
        print(f"delta_MB: {delta_MB}")
        if abs(delta_MB) > 50:  # >5MB threshold
            self._mem_total_leaked_MB += delta_MB

            # Only warn periodically to avoid log spam
            if abs(self._mem_total_leaked_MB) > 100:
                logger.warning(
                    f"⚠️ Memory leak detected: +{delta_MB:.1f}MB (total: {self._mem_total_leaked_MB:.1f}MB) "
                    f"after {self._mem_call_count} calls"
                )

                # Do detailed analysis once when significant leaks detected
                try:
                    
                    torch.cuda.empty_cache()

                    # Get snapshot
                    snapshot = torch.cuda.memory._snapshot()
                    active = [s for s in snapshot.segments if s.is_active]

                    if active:
                        largest = max(active, key=lambda s: s.size)
                        logger.warning(f"Largest allocation: {largest.size / (1024**2):.1f}MB")

                        if hasattr(largest, "frames") and largest.frames:
                            logger.warning(f"Frames: {largest.frames}")
                            for frame in largest.frames[-3:]:
                                if not "site-packages/torch/" in frame.filename:
                                    logger.warning(f"  {frame.filename}:{frame.line}")
                        else:
                            logger.warning("No frames found for largest allocation")
                    else:
                        logger.warning("No active allocations found")
                except Exception as e:
                    logger.warning(f"Error analyzing memory: {e}")
                    breakpoint
                finally:
                    # Disable tracking
                    torch.cuda.memory._record_memory_history(enabled=None)
                    self._mem_last_warning = self._mem_call_count

        return activations_HSPD

    def __compute_model_activations_HSPD(
        self,
        model: HookedTransformer,
        sequence_HS: torch.Tensor,
    ) -> torch.Tensor:
        """Compute activations by running the model. No caching involved."""
        # If I uncomment this, memory usage is stable
        # return torch.rand(*sequence_HS.shape, self._num_hookpoints, model.cfg.d_model)

        # this seems to be the culprit: memory usage goes up and down seemingly randomly and eventally OOMs
        with torch.no_grad():
            _, cache = model.run_with_cache(
                sequence_HS,
                names_filter=self._names_filter,
                stop_at_layer=self._layer_to_stop_at,
            )
        # cache[name] is shape HSD, so stacking on dim 2 = HSPD
        activations_HSPD = torch.stack(
            [cache[name].clone() for name in self._hookpoints], dim=2
        )  # adds hookpoint dim (P)
        del cache
        return activations_HSPD

    def _get_model_activations_HSPD(
        self,
        model: HookedTransformer,
        sequence_HS: torch.Tensor,
    ) -> torch.Tensor:
        # Check if we can load from cache
        if self._cache:
            cache_key = self._cache.get_cache_key(model, sequence_HS)
            activations_HSPD = self._cache.load_activations(cache_key, sequence_HS.device)

            if activations_HSPD is None:
                activations_HSPD = self._compute_model_activations_HSPD(model, sequence_HS)
                self._cache.save_activations(cache_key, activations_HSPD)

            return activations_HSPD

        # Compute activations if not cached or cache loading failed
        activations_HSPD = self._compute_model_activations_HSPD(model, sequence_HS)

        return activations_HSPD

    def get_activations_HSMPD(
        self,
        sequence_HS: torch.Tensor,
    ) -> torch.Tensor:
        activations = [self._get_model_activations_HSPD(model, sequence_HS) for model in self._llms]
        activations_HSMPD = torch.stack(activations, dim=2)
        return activations_HSMPD


def _get_layer(hookpoint: str) -> int:
    if "blocks" not in hookpoint:
        raise NotImplementedError(
            f'Hookpoint "{hookpoint}" is not a blocks hookpoint, cannot determine layer, (but feel free to add this functionality!)'
        )
    assert hookpoint.startswith("blocks.")
    return int(hookpoint.split(".")[1])
