"""Stage 01: reversible UTF-8 byte tokenization. No external tokenizer required."""


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        bytes = []

        if add_bos:
            bytes.append(self.bos_id)

        bytes.extend(text.encode("utf-8"))

        return bytes


    def decode(self, ids: list[int]) -> str:
        start_idx = 0
        end_idx = len(ids) - 1

        if len(ids) and ids[start_idx] == self.bos_id:
            start_idx += 1

        if len(ids) and ids[end_idx] == self.eos_id:
            end_idx -= 1

        return bytes(ids[start_idx : end_idx + 1]).decode("utf-8")
