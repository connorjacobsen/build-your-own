import pytest
import torch


@pytest.mark.examples
def test_atomic_allocation_and_refcounts(api, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=3, block_size=4)
    blocks = pool.allocate(2)
    assert len(set(blocks)) == 2 and pool.n_free == 1
    before = list(pool.refs)
    with pytest.raises(MemoryError):
        pool.allocate(2)
    assert pool.refs == before and pool.n_free == 1, "Failed allocation must leave ownership intact"
    pool.retain([blocks[0]])
    pool.release(blocks)
    assert pool.n_free == 2
    pool.release([blocks[0]])
    pool.check()
    assert pool.n_free == 3
    with pytest.raises(ValueError):
        pool.release([blocks[0]])


@pytest.mark.examples
def test_slot_mapping_storage_and_shared_write_guard(api, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=3, block_size=4)
    allocated = pool.allocate(3)
    table = [allocated[2], allocated[0], allocated[1]]
    blocks, offsets = pool.slots(table, 3, 7)
    assert list(blocks) == [table[0]] + [table[1]] * 4 + [table[2]] * 2
    assert list(offsets) == [3, 0, 1, 2, 3, 0, 1]
    k, v = (
        torch.randn(7, cfg.n_kv_heads, cfg.head_dim),
        torch.randn(7, cfg.n_kv_heads, cfg.head_dim),
    )
    pool.write(0, table, 0, k, v)
    actual = pool.read(0, table, 7)
    torch.testing.assert_close(actual[0], k)
    torch.testing.assert_close(actual[1], v)
    assert pool.storage_bytes == cfg.kv_bytes_per_token() * 12
    pool.retain([table[0]])
    with pytest.raises(ValueError):
        pool.write(0, table, 0, k[:1], v[:1])
    pool.release([table[0]])
    pool.release(table)
    pool.check()


@pytest.mark.examples
@torch.inference_mode()
def test_real_paged_execution_with_shuffled_blocks(api, model, oracle, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=8, block_size=2)
    blocks = pool.allocate(8)
    ta, tb = [blocks[i] for i in [5, 1, 7, 3]], [blocks[i] for i in [0, 6, 2, 4]]
    a, b = [1, 3, 5, 7, 9], [2, 4, 6, 8, 10, 12, 14]
    first = model.forward_paged_batch([(a[:3], ta, 0), (b[:4], tb, 0)], pool)
    second = model.forward_paged_batch([(a[3:], ta, 3), (b[4:], tb, 4)], pool)
    for ids, x, y in zip([a, b], first, second):
        expected, _ = oracle(ids)
        torch.testing.assert_close(torch.cat([x, y]), expected, atol=2e-6, rtol=2e-5)
    pool.release(ta + tb)
    pool.check()


@pytest.mark.extended
@torch.inference_mode()
def test_physical_pages_are_the_source_of_truth(api, model, oracle, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=4, block_size=2)
    table = list(reversed(pool.allocate(4)))
    model.forward_paged_batch([([1, 2, 3, 4], table, 0)], pool)
    _, reference_past = oracle([1, 2, 3, 4])
    for layer in range(cfg.n_layers):
        pool.v[layer, table[:2]] += 3
    edited = [(k, v + 3) for k, v in reference_past]
    actual = model.forward_paged_batch([([5, 6], table, 4)], pool)[0]
    expected, _ = oracle([5, 6], edited)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)


@pytest.mark.extended
@torch.inference_mode()
def test_paged_projection_work_only_new_rows(api, model, cfg):
    pool = api.cache.BlockPool(cfg, num_blocks=8, block_size=4)
    a, b = pool.allocate(4), pool.allocate(4)
    rows = []
    hook = model.layers[0].q.register_forward_pre_hook(
        lambda module, args: rows.append(args[0].shape[0])
    )
    try:
        model.forward_paged_batch([([1, 2, 3], a, 0), ([4, 5], b, 0)], pool)
        model.forward_paged_batch([([6], a, 3), ([7, 8], b, 2)], pool)
    finally:
        hook.remove()
    assert rows == [5, 3], "One packed projection of only new rows per call is required"
