"""Stages 06/08: implement storage, ownership, sharing, and eviction."""

import torch


class BlockPool:
    def __init__(self, cfg, num_blocks=64, block_size=8, device="cpu", dtype=torch.float32):
        raise NotImplementedError("Stage 06: allocate K/V tensors and ownership metadata")

    @property
    def n_free(self):
        raise NotImplementedError("Stage 06: free physical capacity")

    @property
    def storage_bytes(self):
        raise NotImplementedError("Stage 06: persistent K plus V bytes")

    def allocate(self, count):
        raise NotImplementedError("Stage 06: atomic allocation")

    def retain(self, blocks):
        raise NotImplementedError("Stage 06: acquire ownership")

    def release(self, blocks):
        raise NotImplementedError("Stage 06: release ownership exactly once")

    def slots(self, table, start, count):
        raise NotImplementedError("Stage 06: logical-to-physical addresses")

    def write(self, layer, table, start, k, v):
        raise NotImplementedError("Stage 06: scatter new K/V; shared pages are immutable")

    def read(self, layer, table, length):
        raise NotImplementedError("Stage 06: gather only the valid logical prefix")

    def check(self):
        raise NotImplementedError("Stage 06: assert ownership and free-list consistency")

    def copy_on_write(self, table, logical_block):
        raise NotImplementedError("Stage 08: private copy before changing a shared block")


class PrefixCache:
    def __init__(self, pool):
        raise NotImplementedError("Stage 08: prefix identity and retained ownership")

    def acquire(self, prompt):
        raise NotImplementedError("Stage 08: retain a reusable full-block prefix")

    def publish(self, prompt, table, computed):
        raise NotImplementedError("Stage 08: publish computed full prompt blocks only")

    def make_room(self, needed):
        raise NotImplementedError("Stage 08: evict cache-only entries until capacity fits")

    def clear(self):
        raise NotImplementedError("Stage 08: release cache ownership; keep active owners valid")
