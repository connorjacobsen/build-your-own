import pytest
import torch
from grader.oracle.cache import BlockPool, PrefixCache


def test_allocation_failure_atomic_and_reuse(model):
    pool = BlockPool(model.cfg, num_blocks=3, block_size=4)
    table = pool.allocate(2)
    before = pool.refs.copy(), list(pool.free)
    with pytest.raises(MemoryError):
        pool.allocate(2)
    assert (pool.refs, list(pool.free)) == before
    pool.release(table)
    pool.check()
    assert pool.n_free == 3
    with pytest.raises(ValueError):
        pool.release(table)


def test_copy_on_write(model):
    pool = BlockPool(model.cfg, num_blocks=2, block_size=4)
    parent = pool.allocate(1)
    pool.k[:, parent[0]].fill_(3)
    child = parent.copy()
    pool.retain(child)
    new = pool.copy_on_write(child, 0)
    assert new != parent[0]
    assert torch.all(pool.k[:, new] == 3)
    pool.k[:, new].fill_(9)
    assert torch.all(pool.k[:, parent[0]] == 3)
    pool.release(parent + child)
    pool.check()


def test_prefix_context_identity_and_live_eviction(model):
    pool = BlockPool(model.cfg, num_blocks=2, block_size=2)
    prefix = PrefixCache(pool)
    table = pool.allocate(2)
    prefix.publish([1, 2, 3, 4], table, 4)
    assert prefix.acquire([9, 2, 3, 4, 5]) == []
    hit = prefix.acquire([1, 2, 3, 4, 5])
    assert hit == table
    assert not prefix.make_room(1)
    pool.release(table)
    assert not prefix.make_room(1)
    pool.release(hit)
    assert prefix.make_room(2)
    assert not prefix.entries
    pool.check()
