import torch
from fastapi.testclient import TestClient
from grader.oracle.sampling import sample, SamplingParams
from grader.oracle.server import create_app
from grader.oracle.tokenizer import ByteTokenizer


def test_tokenizer_roundtrip():
    t = ByteTokenizer()
    for text in ["", "hello", "naïve 🦙 中文"]:
        assert t.decode(t.encode(text)) == text


def test_sampling_boundaries():
    logits = torch.tensor([1.0, 3.0, 2.0])
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        assert sample(logits, SamplingParams(temperature=1, top_k=1), g) == 1
        assert sample(logits, SamplingParams(temperature=1, top_p=0.1), g) == 1
    assert sample(logits) == 1


def test_local_service(model):
    app = create_app(model)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        payload = {"prompt": "hello", "max_new_tokens": 4}
        a = client.post("/generate", json=payload)
        b = client.post("/generate", json=payload)
        assert a.status_code == b.status_code == 200
        assert a.json()["token_ids"] == b.json()["token_ids"]
        assert len(a.json()["token_ids"]) == 4
        assert (
            client.post("/generate", json={"prompt": "x", "max_new_tokens": 0}).json()["token_ids"]
            == []
        )
        assert client.post("/generate", json={"prompt": "x" * 600}).status_code == 400
        assert (
            client.post("/generate", json={"prompt": "x", "max_new_tokens": -1}).status_code == 422
        )
    assert app.state.engine.pool.n_free == app.state.engine.pool.num_blocks
