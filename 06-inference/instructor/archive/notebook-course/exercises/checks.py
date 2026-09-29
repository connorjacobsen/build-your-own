"""Behavioral checks for learner functions; independent expectations where practical."""

import math
import torch
from nanovllm_course import make_model, TinyConfig
from nanovllm_course.sampling import generate_naive


def check(name, module=None):
    if module is None:
        from exercises import implementation as module
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = make_model(cfg=TinyConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=64))
    if name == "attention":
        q, k, v = torch.randn(3, 4, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8)
        expected = torch.empty_like(q)
        for t in range(3):
            for h in range(4):
                scores = k[: 5 + t, h // 2] @ q[t, h] / math.sqrt(8)
                expected[t, h] = scores.softmax(0) @ v[: 5 + t, h // 2]
        actual = module.causal_attention(q, k, v, 4)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        changed = v.clone()
        changed[5:] += 1000
        torch.testing.assert_close(module.causal_attention(q, k, changed, 4)[0], actual[0])
    elif name == "memory":
        assert module.kv_bytes(32, 8192, 8, 128, 2) == 2**30
        assert module.kv_bytes(2, 0, 2, 16, 4) == 0
    elif name == "rope":
        x = torch.randn(5, 2, 8)
        positions = torch.arange(3, 8)
        result = module.apply_rope(x, positions)
        torch.testing.assert_close(result.square().sum(-1), x.square().sum(-1))
        torch.testing.assert_close(module.apply_rope(x[:1], torch.tensor([0])), x[:1])
        expected = torch.empty_like(x)
        for t, p in enumerate(positions):
            for pair in range(4):
                angle = float(p) * 10000 ** (-(2 * pair) / 8)
                a, b = x[t, :, 2 * pair], x[t, :, 2 * pair + 1]
                expected[t, :, 2 * pair] = a * math.cos(angle) - b * math.sin(angle)
                expected[t, :, 2 * pair + 1] = a * math.sin(angle) + b * math.cos(angle)
        torch.testing.assert_close(result, expected)
    elif name == "sampling":
        logits = torch.tensor([0.15, 0.5, 0.05, 0.3]).log()
        torch.testing.assert_close(
            module.nucleus_probabilities(logits, 0.6), torch.tensor([0.0, 0.625, 0.0, 0.375])
        )
        torch.testing.assert_close(module.nucleus_probabilities(logits, 1), logits.exp())
        torch.testing.assert_close(
            module.nucleus_probabilities(logits, 0.01), torch.tensor([0.0, 1.0, 0.0, 0.0])
        )
    elif name == "decode":
        calls = []

        class Spy:
            def __call__(self, ids, past=None):
                calls.append(len(ids))
                return model(ids, past)

        expected = generate_naive(model, [256, 8, 2], 5)
        assert module.generate_with_cache(Spy(), [256, 8, 2], 5) == expected
        assert calls == [3, 1, 1, 1, 1], f"Unexpected model input lengths: {calls}"
        assert module.generate_with_cache(Spy(), [1], 0) == []
    elif name == "packing":
        assert module.pack_tokens([[8, 9], [7]], [4, 20]) == ([8, 9, 7], [4, 5, 20], [0, 2, 3])
        assert module.pack_tokens([[1], [2, 3, 4], [5, 6]], [0, 7, 2]) == (
            [1, 2, 3, 4, 5, 6],
            [0, 7, 8, 9, 2, 3],
            [0, 1, 4, 6],
        )
    elif name == "slots":
        assert module.physical_slots([5, 1, 7], 4, 3, 7) == (
            [5, 1, 1, 1, 1, 7, 7],
            [3, 0, 1, 2, 3, 0, 1],
        )
        assert module.physical_slots([2], 4, 4, 0) == ([], [])
        try:
            module.physical_slots([2], 4, 3, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-capacity positions must fail")
    elif name == "paged":
        for p in [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]:
            assert module.run_paged(model, p, 7) == generate_naive(model, p, 7)
        assert module.run_paged(model, [1], 0) == []
    elif name == "scheduler":
        assert module.schedule_round([8, 1, 5], 6, 3) == [3, 1, 2]
        assert module.schedule_round([8, 1, 5], 1, 3) == [1, 0, 0]
        assert module.schedule_round([0, 2, 1], 8, 3) == [0, 2, 1]
        assert module.schedule_round([2, 3], 0, 3) == [0, 0]
    elif name == "prefix":
        assert [module.reusable_prefix_tokens(n, 4) for n in [1, 3, 4, 5, 8, 9]] == [
            0,
            0,
            0,
            4,
            4,
            8,
        ]
    elif name == "engine":
        engine = module.build_engine(
            model, num_blocks=12, block_size=4, token_budget=3, prefill_chunk=2
        )
        engine.add_request("a", [256, 1, 2, 3, 4], 5)
        engine.step()
        engine.add_request("b", [256, 9], 3)
        engine.add_request("cancel", [256, 9], 4)
        engine.cancel("cancel")
        engine.cancel("cancel")
        engine.add_request("zero", [1], 0)
        for _ in range(200):
            if not engine.has_work:
                break
            engine.step()
        assert not engine.has_work, "Engine did not make progress"
        assert engine.requests["a"].output == generate_naive(model, [256, 1, 2, 3, 4], 5)
        assert engine.requests["b"].output == generate_naive(model, [256, 9], 3)
        assert engine.requests["zero"].output == []
        assert engine.requests["cancel"].status == "cancelled"
        assert all(h["tokens"] <= 3 for h in engine.history)
        engine.clear_prefix_cache()
        assert engine.pool.n_free == engine.pool.num_blocks
    else:
        raise ValueError(f"Unknown exercise: {name}")
    print(f"PASS {name}")


NAMES = [
    "attention",
    "memory",
    "rope",
    "sampling",
    "decode",
    "packing",
    "slots",
    "paged",
    "scheduler",
    "prefix",
    "engine",
]
