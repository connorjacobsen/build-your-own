> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 01 — Bytes and the generation boundary

**Build:** `src/toyvllm/tokenizer.py`  
**Prerequisites:** Python sequences, classes, strings, and exceptions.  
**Estimated effort:** 30–60 minutes.  
**Gate:** use the microstage number shown by `uv run course list`.

## The system you are starting

An inference engine is a program that turns requests into generated token sequences while managing computation and memory. It has a mathematical component—the model—and a systems component that decides whose work runs, where state lives, and when resources can be reclaimed. You will build both. The instructor grader is a consumer of your public interfaces, not a library you call to generate answers.

The first boundary is text to token IDs. A model does not directly process Python strings. It consumes integers indexing a vocabulary. Tokenization defines these IDs; an embedding table later maps them to learned vectors. Keep the boundary small now so it remains stable as model execution becomes more complicated.

There are ten cumulative stages. You edit the implementation under `src/toyvllm/`. The instructor owns the tests and oracle. Run the full gate after a stage: it checks that the new behavior works and earlier behavior still works. Example-only runs are useful during development but do not complete a stage.

## Contract

Implement `ByteTokenizer`, preserving the supplied constants:

| Member | Required behavior |
|---|---|
| `bos_id` | Integer 256, beginning-of-sequence marker |
| `eos_id` | Integer 257, end-of-sequence marker |
| `vocab_size` | Integer 258 |
| `encode(text, add_bos=True)` | New list of UTF-8 byte IDs, optionally preceded by BOS |
| `decode(ids)` | Decode IDs in 0–255 as UTF-8 with replacement for invalid byte sequences; ignore IDs outside the byte range |

Both operations must leave caller-owned inputs untouched. The input domain is Python text for encoding and integer sequences for decoding. This stage does not require a learned tokenizer, BPE, external vocabulary files, or model downloads.

Public examples:

| Expression | Expected result |
|---|---|
| `encode("Hi")` | `[256, 72, 105]` |
| `encode("café", add_bos=False)` | `[99, 97, 102, 195, 169]` |
| `encode("")` | `[256]` |
| `encode("", add_bos=False)` | `[]` |
| `decode([256,72,105,257])` | `"Hi"` |
| `decode([255])` | The replacement character `"�"` |

The generation interface is only a boundary at this stage: future callers submit prompt IDs and a maximum output length, and receive generated IDs. You do not implement generation until stage 03.

## Why bytes, and why special IDs?

UTF-8 represents Unicode text as bytes. One character may need multiple bytes: `é` needs two; many emoji need four. Python's `len(text)` counts code points, not UTF-8 bytes and not model tokens. With this tokenizer, ordinary text uses one token per byte. BOS adds another token.

The distinction matters later. A maximum context measured in tokens cannot be validated by counting displayed characters. A model's output can also end halfway through a multibyte character. Your offline decoder must handle invalid sequences; a future streaming decoder would need to retain incomplete bytes until more arrive.

BOS gives even an empty text prompt a valid input position. The final input position produces next-token logits, so an actual empty token list is unsuitable for the generation loop. EOS is a model output convention: it is an ID that a caller may choose to treat as a stopping condition. Merely reserving ID 257 does not mean every future request must stop when it appears.

A production tokenizer often represents common byte sequences with single IDs. That improves sequence efficiency, but adds vocabulary and compatibility details that obscure the first systems lessons. Our tokenizer is deliberately simple and reversible. It is not compatible with arbitrary pretrained checkpoints.

## Your working loop

Start with `uv sync --locked`. Then run:

```bash
uv run course read 1
uv run course check 1 --examples
uv run course check 1
```

The initial failure is expected and must exit nonzero. There is no reference fallback when your method raises `NotImplementedError`. Implement the method, rerun, and use the failure message to refine your understanding. You do not need to edit tests or fill in expected values.

The additional acceptance cases include randomized Unicode strings and input-immutability checks. Their randomness has a recorded seed. You can explore another deterministic case set with `uv run course check 1 --seed 91`. These are local, inspectable instructor tests; they are not falsely presented as a secure hidden-test service.

## Completion and reflection

Explain why `decode(encode(text)) == text` should hold, but `encode(decode(ids)) == ids` need not hold for arbitrary IDs. Special tokens are omitted during decoding, invalid bytes lose information through replacement, and default encoding adds BOS. Round-trip laws need precise domains.

Your result is a small complete component, not an exercise stub that the final engine will bypass. The HTTP frontend will use this exact tokenizer in stage 09.

**Next:** [Stage 02 — A causal transformer](02_model.md). If stuck, ask for one hint with `uv run course hint 1 --level 1`; the CLI never reveals complete solution code.
