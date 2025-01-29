from model_diffing.dataloader.activations import ActivationsDataloader, ActivationsHarvester
from model_diffing.dataloader.token_loader import ToyOverfittingTokenSequenceLoader
from model_diffing.scripts.config_common import LLMConfig
from model_diffing.scripts.llms import build_llms
from model_diffing.utils import get_device


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

    dataloader = ActivationsDataloader(
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=training_batch_size * 4,
        token_sequence_loader=sequence_loader,
        yield_batch_size=training_batch_size,
    )

    sample_activations_batch = next(dataloader.get_shuffled_activations_iterator_BMLD())
    harvested_activation_expected_shape_MLD = (training_batch_size, len(llms), len(layer_indices_to_harvest), d_model)
    assert sample_activations_batch.shape == harvested_activation_expected_shape_MLD
