# 30. Serve a checkpoint you trained

Implement `src/toyvllm/artifacts.py: load_export`. Training and serving must agree on more than tensor shapes: tokenizer identity, normalization, rotary convention, parameter orientation and architecture all matter. The shared `toylm-v1` format records those choices.

Load the trusted local file using `torch.load(weights_only=True, map_location="cpu")`. Require format `toylm-v1` and tokenizer metadata exactly `{kind: utf8-byte, bos_id: 256, eos_id: 257, vocab_size: 258}`. Configuration vocabulary must also be 258. Construct `TinyLM(TinyConfig(**config))` without altering caller CPU RNG state, strict-load `state_dict`, move to the requested device and return in eval mode. See the scaffold signature for exception conventions.

Run `uv run course check 30`. After completing pretraining, export your trained checkpoint and verify dense logits before turning on cache reuse or prefix sharing. The suite integration check compares the same artifact across both courses. Do not assume an arbitrary external Hugging Face checkpoint shares this layout.

Explain: which metadata could be wrong even when every parameter shape matches? Why is inference mode a caller execution choice whereas eval mode is model state?
