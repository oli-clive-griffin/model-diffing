from model_diffing.dataloader.activations import ActivationsHarvester
from model_diffing.dataloader.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.dataloader.token_loader import ToyOverfittingTokenSequenceIterator
from model_diffing.scripts.config_common import LLMConfig, LLMsConfig
from model_diffing.scripts.llms import build_llms
from model_diffing.utils import get_device


def test():
    llms = build_llms(
        LLMsConfig(models=[LLMConfig(name="EleutherAI/pythia-160M", revision="step142000")]),
        cache_dir=".cache",
        device=get_device(),
    )

    sequence_len = 10
    d_model = llms[0].cfg.d_model

    token_sequence_iterator_S = ToyOverfittingTokenSequenceIterator(sequence_len).get_sequence_iterator()
    assert next(token_sequence_iterator_S).shape == (sequence_len,)

    harvesting_batch_size = 4
    shuffled_token_sequence_iterator_BS = batch_shuffle_tensor_iterator_BX(
        tensor_iterator_X=token_sequence_iterator_S,
        shuffle_buffer_size=20,
        yield_batch_size=harvesting_batch_size,
    )
    tokens_batch_expected_shape_BS = (harvesting_batch_size, sequence_len)
    assert next(shuffled_token_sequence_iterator_BS).shape == tokens_batch_expected_shape_BS

    layer_indices_to_harvest = [2, 4, 6]
    token_activations_iterator_MLD = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=layer_indices_to_harvest,
        token_sequence_iterator_BS=shuffled_token_sequence_iterator_BS,
    ).get_token_activations_iterator_MLD()
    harvested_activation_expected_shape_MLD = (len(llms), len(layer_indices_to_harvest), d_model)
    assert next(token_activations_iterator_MLD).shape == harvested_activation_expected_shape_MLD

    training_batch_size = 6
    shuffled_activations_iterator_BMLD = batch_shuffle_tensor_iterator_BX(
        tensor_iterator_X=token_activations_iterator_MLD,
        shuffle_buffer_size=20,
        yield_batch_size=training_batch_size,
    )
    train_batch_expected_shape_BMLD = (training_batch_size, len(llms), len(layer_indices_to_harvest), d_model)
    assert next(shuffled_activations_iterator_BMLD).shape == train_batch_expected_shape_BMLD
