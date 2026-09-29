import random
import pytest


@pytest.mark.examples
def test_utf8_bytes_and_special_ids(api):
    t = api.tokenizer.ByteTokenizer()
    assert (t.bos_id, t.eos_id, t.vocab_size) == (256, 257, 258)
    assert t.encode("Hi") == [256, 72, 105], "encode must prepend BOS, then actual UTF-8 byte IDs"
    assert t.encode("café", add_bos=False) == [99, 97, 102, 195, 169]
    assert t.decode([256, 72, 105, 257]) == "Hi"


@pytest.mark.examples
def test_empty_prompt_and_invalid_utf8(api):
    t = api.tokenizer.ByteTokenizer()
    assert t.encode("") == [256]
    assert t.encode("", add_bos=False) == []
    assert t.decode([]) == ""
    assert t.decode([255]) == "�", "Invalid byte sequences use replacement decoding"


@pytest.mark.extended
def test_unicode_roundtrip_and_no_input_mutation(api, case_seed):
    t = api.tokenizer.ByteTokenizer()
    rng = random.Random(case_seed)
    alphabet = "a\x00\n é中文🦙Ω"
    for _ in range(30):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(40)))
        expected = [256] + list(text.encode("utf-8"))
        assert t.encode(text) == expected
        before = expected.copy()
        assert t.decode(expected) == text
        assert expected == before, "Decoding must not modify caller-owned IDs"
