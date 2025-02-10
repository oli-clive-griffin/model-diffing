from model_diffing.data.model_layer_dataloader import ActivationsHarvester, ScaledModelLayerActivationsDataloader
from model_diffing.data.token_loader import ToyOverfittingTokenSequenceLoader
from model_diffing.scripts.config_common import LLMConfig
from model_diffing.scripts.llms import build_llms
from model_diffing.utils import get_device

# TODO(oli): fixme, this test is slow because it estimates the norm scaling factor.
# Need to find a better way to test the shapes while not actually streaming a large amount of data


def test():
    llms = build_llms(
        [LLMConfig(name="EleutherAI/pythia-160M", revision="step142000")],
        cache_dir=".cache",
        device=get_device(),
        dtype="float32",
    )

    sequence_len = 16
    harvesting_batch_size = 1
    training_batch_size = 8
    d_model = llms[0].cfg.d_model

    sequence_loader = ToyOverfittingTokenSequenceLoader(batch_size=harvesting_batch_size, sequence_length=sequence_len)

    sample_batch = next(sequence_loader.get_sequences_batch_iterator())
    assert sample_batch.shape == (harvesting_batch_size, sequence_len)

    layer_indices_to_harvest = [6]
    activations_harvester = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=layer_indices_to_harvest,
    )

    dataloader = ScaledModelLayerActivationsDataloader(
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=training_batch_size * 4,
        token_sequence_loader=sequence_loader,
        yield_batch_size=training_batch_size,
        device=get_device(),
        n_batches_for_norm_estimate=1,
    )

    sample_activations_batch = next(dataloader.get_shuffled_activations_iterator_BMLD())
    harvested_activation_expected_shape_MLD = (training_batch_size, len(llms), len(layer_indices_to_harvest), d_model)
    assert sample_activations_batch.shape == harvested_activation_expected_shape_MLD
