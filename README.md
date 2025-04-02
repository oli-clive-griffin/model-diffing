# Crosscode

A library for training crosscoders, and by extension, transcoders, SAEs, and other sparse coding models.

### Training Examples:
- Training a multi-layer / multi-model crosscoder
    - With Vanilla L1 loss as in the original crosscoder paper: [here](./crosscode/trainers/l1_crosscoder/run.py)
    - With TopK/BatchTopK/GroupMax: [here](./crosscode/trainers/topk_crosscoder/run.py)
    - With JumpReLU according to Anthropic's January 2025 update: [here](./crosscode/trainers/jan_update_crosscoder/run.py)
    - According to Anthropic's February 2025 model diffing update
        - With JumpReLU (as in the paper): [here](./crosscode/trainers/feb_update_diffing_crosscoder/run_jumprelu.py)
        - With TopK/: [here](./crosscode/trainers/feb_update_diffing_crosscoder/topk_trainer.py)
    - (Any of these can also be an SAE by running on one LLM only and setting `hookpoints` to a single layer)
- Training a skip transcoder
    - With TopK/BatchTopK/GroupMax: [here](./crosscode/trainers/skip_transcoder/run.py)
- Training a cross-layer transcoder
    - With L1 loss: [here](./crosscode/trainers/l1_crosslayer_trancoder/run.py)


### Models:
- Acausal crosscoder (across models and/or layers): [here](./crosscode/models/acausal_crosscoder.py)
- Cross-layer transcoder: [here](./crosscode/models/crosslayer_transcoder.py)
- (Sketch of) The compound cross-layer trancoder from "[Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)": [here](./crosscode/models/compound_clt.py)
- Sparse Autoencoder / Transcoder: [here](./crosscode/models/sae.py)
