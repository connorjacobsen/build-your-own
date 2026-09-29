"""A reversible UTF-8 byte tokenizer: no vocabulary files or model downloads."""


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        return ([self.bos_id] if add_bos else []) + list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")
