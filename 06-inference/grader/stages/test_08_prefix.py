import pytest
import torch


@pytest.mark.examples
def test_full_prefix_identity_last_logit_and_active_eviction(api, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=2, block_size=2)
    cache = api.cache.PrefixCache(pool)
    table = pool.allocate(2)
    cache.publish([1, 2, 3, 4], table, 4)
    assert cache.acquire([9, 2, 3, 4, 5]) == [], (
        "Identical local block tokens do not imply identical context"
    )
    exact = cache.acquire([1, 2, 3, 4])
    assert exact == table[:1], "An exact-boundary prompt must retain work for final logits"
    pool.release(exact)
    hit = cache.acquire([1, 2, 3, 4, 5])
    assert hit == table
    pool.release(table)
    assert not cache.make_room(1), "Never evict a block still owned by an active request"
    pool.release(hit)
    assert cache.make_room(2) and pool.n_free == 2
    pool.check()


@pytest.mark.extended
def test_only_computed_full_blocks_publish_and_clear_preserves_owners(api, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=3, block_size=4)
    cache = api.cache.PrefixCache(pool)
    table = pool.allocate(3)
    cache.publish(list(range(10)), table, 6)
    hit = cache.acquire(list(range(10)))
    assert len(hit) == 1, "A partially computed block must not be published"
    cache.clear()
    assert pool.n_free == 0 and pool.refs[hit[0]] == 2
    pool.release(hit)
    pool.release(table)
    assert pool.n_free == 3


@pytest.mark.examples
def test_copy_on_write_and_failure_atomicity(api, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=2, block_size=4)
    parent = pool.allocate(1)
    pool.k[:, parent[0]].fill_(3)
    child = parent.copy()
    pool.retain(child)
    new = pool.copy_on_write(child, 0)
    assert new != parent[0]
    assert torch.all(pool.k[:, new] == 3)
    pool.k[:, new, 2].fill_(9)
    assert torch.all(pool.k[:, parent[0], 2] == 3)
    pool.retain(parent)
    before = parent.copy(), pool.refs.copy()
    with pytest.raises(MemoryError):
        pool.copy_on_write(parent, 0)
    assert (parent, pool.refs) == before
    pool.release(parent + parent + child)
    pool.check()


@pytest.mark.extended
@pytest.mark.parametrize("prompt_length", [8, 9])
def test_prefix_reuse_skips_real_projection_work(api, model, prompt_length):
    engine = api.engine.Engine(
        model, num_blocks=12, block_size=4, token_budget=8, prefill_chunk=4, prefix_caching=True
    )
    projected = []
    hook = model.layers[0].q.register_forward_pre_hook(
        lambda module, args: projected.append(args[0].shape[0])
    )
    prompt = list(range(prompt_length))
    try:
        engine.add_request("cold", prompt, 4)
        engine.run()
        cold = sum(projected)
        projected.clear()
        engine.add_request("warm", prompt, 4)
        engine.run()
        warm = sum(projected)
    finally:
        hook.remove()
    reused = ((prompt_length - 1) // 4) * 4
    assert cold == prompt_length + 3 and cold - warm == reused, (
        "Prefix hits must skip model work, not only increment a hit counter"
    )
    assert engine.requests["warm"].cached_tokens == reused
    assert engine.requests["cold"].output == engine.requests["warm"].output
    engine.clear_prefix_cache()
    assert engine.pool.n_free == 12


@pytest.mark.extended
def test_prefix_pressure_recycles_capacity(api, model):
    engine = api.engine.Engine(
        model, num_blocks=4, block_size=4, token_budget=3, prefill_chunk=2, prefix_caching=True
    )
    for i in range(8):
        engine.add_request(str(i), [i] * 9, 5)
        engine.run()
        engine.pool.check()
    engine.clear_prefix_cache()
    assert engine.pool.n_free == 4
