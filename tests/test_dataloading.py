from crosscoding.data.model_hookpoint_dataloader import (
    ActivationsHarvester,
    ScaledModelHookpointActivationsDataloader,
)
from crosscoding.data.token_hookpoint_dataloader import SlidingWindowScaledActivationsDataloader
from crosscoding.data.token_loader import ToyOverfittingTokenSequenceLoader
from crosscoding.trainers.config_common import LLMConfig
from crosscoding.trainers.llms import build_llms
from crosscoding.utils import get_device

# TODO(oli): fixme, this test is slow because it estimates the norm scaling factor.
# Need to find a better way to test the shapes while not actually streaming a large amount of data


def test_MP():
    llms = build_llms(
        [LLMConfig(name="EleutherAI/pythia-160M", revision="step142000")],
        cache_dir=".cache",
        device=get_device(),
        inferenced_type="float32",
    )

    sequence_len = 16
    harvesting_batch_size = 1
    training_batch_size = 8
    d_model = llms[0].cfg.d_model

    sequence_loader = ToyOverfittingTokenSequenceLoader(
        batch_size=harvesting_batch_size,
        sequence_length=sequence_len,
    )

    sample_batch = next(sequence_loader.get_sequences_batch_iterator())
    assert sample_batch.tokens_HS.shape == (harvesting_batch_size, sequence_len)

    hookpoints = ["blocks.6.hook_resid_post"]
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
    )

    dataloader = ScaledModelHookpointActivationsDataloader(
        activations_harvester=activations_harvester,
        token_sequence_loader=sequence_loader,
        yield_batch_size_B=training_batch_size,
        n_tokens_for_norm_estimate=1,
    )

    sample_activations_batch_BMPD = next(dataloader.get_activations_iterator_BMPD())

    assert sample_activations_batch_BMPD.shape == (
        training_batch_size,  # B
        len(llms),  # M
        len(hookpoints),  # P
        d_model,  # D
    )


def test_TPD():
    llms = build_llms(
        [LLMConfig(name="EleutherAI/pythia-160M", revision="step142000")],
        cache_dir=".cache",
        device=get_device(),
        inferenced_type="float32",
    )

    sequence_len = 16
    harvesting_batch_size = 1
    training_batch_size = 8
    d_model = llms[0].cfg.d_model
    window_size = 2

    sequence_loader = ToyOverfittingTokenSequenceLoader(
        batch_size=harvesting_batch_size,
        sequence_length=sequence_len,
    )

    sample_batch = next(sequence_loader.get_sequences_batch_iterator())
    assert sample_batch.tokens_HS.shape == (harvesting_batch_size, sequence_len)

    hookpoints = ["blocks.6.hook_resid_post"]
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
    )

    dataloader = SlidingWindowScaledActivationsDataloader(
        activations_harvester=activations_harvester,
        token_sequence_loader=sequence_loader,
        yield_batch_size=training_batch_size,
        n_tokens_for_norm_estimate=1,
        window_size=window_size,
    )

    sample_activations_batch_BTPD = next(dataloader.get_activations_iterator_BTPD())

    assert sample_activations_batch_BTPD.shape == (
        training_batch_size,  # B
        window_size,  # T
        len(hookpoints),  # P
        d_model,  # D
    )
