"""Prove that common output-correct but structurally wrong implementations fail."""

from types import SimpleNamespace
import pytest
from grader.oracle import make_model, TinyConfig
from grader.oracle import sampling, cache, engine
from grader.stages.test_04_kv_cache import test_cache_reuses_work_not_just_outputs as check_reuse
from grader.stages.test_05_batching import (
    test_projections_are_packed_not_sequential_calls as check_packing,
)
from grader.stages.test_06_paging import test_physical_pages_are_the_source_of_truth as check_pages
from grader.stages.test_08_prefix import (
    test_prefix_reuse_skips_real_projection_work as check_prefix,
)


def matched():
    cfg = TinyConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=64)
    return make_model(seed=101, cfg=cfg), make_model(seed=101, cfg=cfg), cfg


def test_rejects_dense_recomputation_disguised_as_cached_generation():
    model, oracle, _ = matched()
    bad = SimpleNamespace(sampling=SimpleNamespace(generate_cached=sampling.generate_naive))
    with pytest.raises(AssertionError, match="Feed back only"):
        check_reuse(bad, model, oracle)


def test_rejects_sequential_execution_disguised_as_packing():
    model, _, _ = matched()

    def sequential(sequences, pasts=None):
        results = [
            model(ids, past) for ids, past in zip(sequences, pasts or [None] * len(sequences))
        ]
        return [x[0] for x in results], [x[1] for x in results]

    model.forward_batch = sequential
    with pytest.raises(AssertionError, match="packed six-row"):
        check_packing(model)


def test_rejects_ignoring_physical_block_indirection(monkeypatch):
    model, oracle, cfg = matched()
    original = cache.BlockPool.slots
    monkeypatch.setattr(
        cache.BlockPool,
        "slots",
        lambda self, table, start, count: original(self, list(range(len(table))), start, count),
    )
    with pytest.raises(AssertionError):
        check_pages(SimpleNamespace(cache=cache), model, oracle, cfg)


def test_rejects_no_reuse_even_with_correct_outputs():
    model, _, _ = matched()

    def uncached(*args, **kwargs):
        kwargs["prefix_caching"] = False
        return engine.Engine(*args, **kwargs)

    with pytest.raises(AssertionError, match="skip model work"):
        check_prefix(SimpleNamespace(engine=SimpleNamespace(Engine=uncached)), model, 9)
