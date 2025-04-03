# Crosscode

A library for training crosscoders, and by extension, transcoders, SAEs, and other sparse coding models.


### Examples:
- Training a multi-layer / multi-model crosscoder
    - With Vanilla L1 loss as in the original [crosscoder paper](https://transformer-circuits.pub/2024/crosscoders/index.html): [here](./crosscode/trainers/l1_crosscoder/run.py)
    - With TopK/BatchTopK/GroupMax: [here](./crosscode/trainers/topk_crosscoder/run.py)
    - With JumpReLU according to Anthropic's [January 2025 update](https://transformer-circuits.pub/2025/january-update/index.html): [here](./crosscode/trainers/jan_update_crosscoder/run.py)
    - According to Anthropic's February 2025 [model diffing update](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html)
        - With JumpReLU (as in the paper): [here](./crosscode/trainers/feb_update_diffing_crosscoder/run_jumprelu.py)
        - With TopK/BatchTopK/GroupMax: [here](./crosscode/trainers/feb_update_diffing_crosscoder/topk_trainer.py)
    - (Any of these can also be an SAE by running on one LLM only and setting `hookpoints` to a single layer)
- Training a cross-layer (skip) transcoder
    - With L1 loss: [here](./crosscode/trainers/l1_crosslayer_trancoder/run.py)
    - With TopK/BatchTopK/GroupMax: [here](./crosscode/trainers/topk_cross_layer_transcoder/run.py)


## Key terms: 
- **crosscoding dimensions**: the dimensions over which the crosscoder is applied.
    - e.g. in a cross-layer crosscoder, the crosscoding dimensions are `(layers,)`
    - e.g. in a cross-model crosscoder, the crosscoding dimensions are `(models,)`
    - e.g. in a cross-model, cross-layer crosscoder, the crosscoding dimensions are `(models, layers)`
- **hookpoints**: the hookpoints at which activations are harvested.
- **latents**: the hidden activations of the crosscoder/transcoder.
- **topk-style**: blanket term for TopK, BatchTopK, and GroupMax activations. Lumped together as they are all trained the same way.
- **Jan Update**: the "[January 2025 update](https://transformer-circuits.pub/2025/january-update/index.html)" describing a specific jumprelu loss.
- **Feb Update**: the "[February 2025 model diffing update](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html)" describing a technique for improving model-diffing crosscoder training with shared latents.

## Conventions:

#### Models, Trainers, Loss Functions.
All sparse coding models are abstracted over activation functions, and losses are handled by trainers. This is nice because different training schemes are usually a combination of (activation function, loss function, some hyperparameters, some scheduling) and this way we put all of that in the trainer class in a type-safe way.

#### Tensor Shapes
This library makes extensive use of "[shape suffixes](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd)" and einops to attempt to make the quite complex and varying tensor shapes a bit more manageable. Usually shapes are usually: 
- `B`: Batch
- `M`: Models
- `P`: Hookpoints (for example, different layers of the residual stream)
- `L`: Latents (aka "features")
- `D`: Model Dimension (aka `d_model`)
- `X`: an arbitrary number of crosscoding dimensions (usually 0, 1, or 2 of them, such as `(n_models, n_layers)`)

Shape suffixes should be interpretted in PascalCase, where lowercase denotes a more specific version of a shape. For example, [here](./crosscode/models/base_crosscoder.py#L51) we have `_W_enc_XiDiL` which means shape ("**I**nput **X**rosscoding dims, **I**nput **D**_model, **L**atents).

#### Dataloading
We currently only have one dataloader type, and handle the reshaping of activations for a given model / training scheme in the trainer classes. Once again trying to keep most of the complexity in the same place.

We currently harvest activations from the LLM(s) at training time. You can cache activations to disk to avoid re-harvesting them in subsequent runs. This, however is probably the least-developed part of the library.


## Structure

The library is structured roughly as follows:
**[`crosscode.models`](./crosscode/models):**
    - [`BaseCrosscoder`](./crosscode/models/base_crosscoder.py): Generic base class for all crosscoding models. It's allowed to have different input and output [crosscoding dimensions](#key-terms) and `d_model`s, and meant to be subclassed in a way that concretifies the dimensions.
        - For example, [`CrossLayerTranscoder`](./crosscode/models/crosslayer_transcoder.py) is a subclass of BaseCrosscoder that concretifies the input crosscoding dimensions to be `(),` and the output dimensions to be `(n_layers,)`.
    - [`ModelHookpointAcausalCrosscoder`](./crosscode/models/acausal_crosscoder.py): An acausal crosscoder that can be applied across multiple models / layers.
        - with `n_layers = 1` and `n_models=2`, it's a normal model diffing crosscoder.
        - with `n_layers > 1` and `n_models=1`, it's a normal cross-layer acausal transcoder.
        - with `n_layers > 1` and `n_models > 1`, it's a cross-layer cross-model acausal transcoder (???).
    - [`CrossLayerTranscoder`](./crosscode/models/cross_layer_transcoder.py): A cross-layer acausal transcoder 
        - [`CompoundCrossLayerTranscoder`](./crosscode/models/compound_clt.py): A wrapper around a list of CrossLayerTranscoder that applies them in parallel, as described in the "[Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)" paper.

**[`crosscode.models.activations`](./crosscode/models/activations):**
A collection of activation functions that can be used with the model classes.

**[`crosscode.models.initialization`](./crosscode/models/initialization):**
A collection of `InitStrategy`s for initializing crosscoder weights.
    - [`InitStrategy`](./crosscode/models/initialization/init_strategy.py): A base class for all initialization strategies.
    - [`AnthropicTransposeInit`](./crosscode/models/initialization/anthropic_transpose.py): Initializes the weights of a ModelHookpointAcausalCrosscoder using the "Anthropic Transpose" method.
    - [`IdenticalLatentsInit`](./crosscode/models/initialization/diffing_identical_latents.py): Initializes the weights of a ModelHookpointAcausalCrosscoder such that the first `n_shared_latents` are identical for all models.
    - [`JanUpdateInit`](./crosscode/models/initialization/jan_update_init.py): Initializes the weights of a ModelHookpointAcausalCrosscoder using the method described in the "[January 2025 update](https://transformer-circuits.pub/2025/january-update/index.html)" paper.
    - Theres's some other random initialization strategies in here that are more speculative.

**[`crosscode.trainers`](./crosscode/trainers):**
(The trainers make extensive use of Inheritance which I really don't like. I might refactor this to use composition instead)
- [`BaseTrainer`](./crosscode/trainers/base_trainer.py): Training boilerplate. Gradient accumulation, logging, optimizer, lr scheduler, etc.
    - [`BaseCrossLayerTranscoderTrainer`](./crosscode/trainers/base_crosslayer_transcoder_trainer.py): A trainer for CrossLayerTranscoder models, handles dataloading, splitting activations into input and output layers.
        - [`L1CrossLayerTranscoderTrainer`](./crosscode/trainers/l1_crosslayer_trancoder/trainer.py): Trains a CrossLayerTranscoder with L1 loss.
        - [`TopkCrossLayerTranscoderTrainer`](./crosscode/trainers/topk_cross_layer_transcoder/trainer.py): Trains a CrossLayerTranscoder with TopK loss.
- [`BaseModelHookpointAcausalTrainer`](./crosscode/trainers/base_acausal_trainer.py): A trainer for ModelHookpointAcausalCrosscoder models.
    - [`TopkStyleAcausalCrosscoderTrainer`](./crosscode/trainers/topk_crosscoder/trainer.py): A trainer for TopK style models.
    - [`BaseFebUpdateDiffingTrainer`](./crosscode/trainers/base_diffing_trainer.py): an extension of BaseModelHookpointAcausalTrainer that implements shared latents as in the "[February 2025 model diffing update](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html)"
        - [`JumpReLUFebUpdateDiffingTrainer`](./crosscode/trainers/feb_update_diffing_crosscoder/jumprelu_trainer.py)
        - [`TopkFebUpdateDiffingTrainer`](./crosscode/trainers/feb_update_diffing_crosscoder/topk_trainer.py)


**[`crosscode.data`](./crosscode/data):**
data loading via harvesting LLM activations on text.
- [`ActivationsDataloader`](./crosscode/data/activations_dataloader.py): Dataloader for activations. Supports harvesting for multiple models and multiple hookpoints.
    - [`TokenSequenceLoader`](./crosscode/data/token_loader.py): Used by ActivationsDataloader to load sequences of tokens from huggingface and chunk them into batches for activations harvesting. Can shuffle across sequences.
    - [`ActivationHarvester`](./crosscode/data/activation_harvester.py): Used by ActivationsDataloader to harvest LLM activations on those sequences.
        - [`ActivationCache`](./crosscode/data/activation_cache.py): Used by ActivationsHarvester to cache activations to disk (if enabled).

