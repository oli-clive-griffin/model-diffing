# Crosscode

A library for trainnig crosscoders, and by extension, transcoders, SAEs, and other sparse coding models.

Examples:
- Training a multi-layer / multi-model crosscoder
    - [With Vanilla L1 loss](./crosscode/trainers/l1_crosscoder/run.py)
    - [With TopK/BatchTopK/GroupMax](./crosscode/trainers/topk_crosscoder/run.py)
    - [With JumpReLU according to Anthropic's January 2025 update](./crosscode/trainers/jan_update_crosscoder/run.py)
    - (Any of these can also be an SAE by running on one LLM only and setting `hookpoints` to a single layer)
- Training a skip transcoder
    - [With TopK/BatchTopK/GroupMax](./crosscode/trainers/skip_transcoder/run.py)
- Training a cross-layer transcoder
    - [With L1 loss](./crosscode/trainers/l1_crosslayer_trancoder/run.py)
