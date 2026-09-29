"""Paged storage and ownership. A physical block stores all layers' K and V."""

from collections import OrderedDict, deque
import torch


class BlockPool:
    def __init__(self, cfg, num_blocks=64, block_size=8, device="cpu", dtype=torch.float32):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("positive block count and size required")
        self.num_blocks, self.block_size = num_blocks, block_size
        shape = (cfg.n_layers, num_blocks, block_size, cfg.n_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.refs = [0] * num_blocks
        self.free = deque(range(num_blocks))

    @property
    def n_free(self):
        return len(self.free)

    @property
    def storage_bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    def allocate(self, count):
        if count < 0:
            raise ValueError("negative allocation")
        if count > self.n_free:
            raise MemoryError("insufficient physical blocks; allocation is atomic")
        blocks = [self.free.popleft() for _ in range(count)]
        for block in blocks:
            assert self.refs[block] == 0
            self.refs[block] = 1
        return blocks

    def retain(self, blocks):
        for block in blocks:
            if self.refs[block] <= 0:
                raise ValueError("cannot retain an unowned block")
            self.refs[block] += 1

    def release(self, blocks):
        for block in blocks:
            if self.refs[block] <= 0:
                raise ValueError("double free")
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)

    def slots(self, table, start, count):
        if start < 0 or count < 0 or start + count > len(table) * self.block_size:
            raise ValueError("logical positions are outside the block table")
        positions = torch.arange(start, start + count, device=self.k.device)
        physical = torch.tensor(table, device=self.k.device, dtype=torch.long)
        return physical[positions // self.block_size], positions % self.block_size

    def write(self, layer, table, start, k, v):
        blocks, offsets = self.slots(table, start, len(k))
        # Shared blocks are immutable. A caller must COW before changing a shared tail.
        if any(self.refs[b] != 1 for b in set(blocks.tolist())):
            raise ValueError("writing unowned/shared blocks; copy on write first")
        self.k[layer, blocks, offsets] = k
        self.v[layer, blocks, offsets] = v

    def read(self, layer, table, length):
        blocks, offsets = self.slots(table, 0, length)
        return self.k[layer, blocks, offsets], self.v[layer, blocks, offsets]

    def copy_on_write(self, table, logical_block):
        old = table[logical_block]
        if self.refs[old] == 1:
            return old
        new = self.allocate(1)[0]  # Failure leaves old table and references unchanged.
        self.k[:, new].copy_(self.k[:, old])
        self.v[:, new].copy_(self.v[:, old])
        table[logical_block] = new
        self.release([old])
        return new

    def check(self):
        free = list(self.free)
        assert len(free) == len(set(free))
        assert set(free) == {i for i, ref in enumerate(self.refs) if ref == 0}
        assert all(ref >= 0 for ref in self.refs)


class PrefixCache:
    """Exact prefix keys and LRU eviction. Scoped to one immutable model/engine.

    Each entry owns one reference. Active requests own additional references.
    This ownership convention differs from vLLM's cached free-list convention.
    """

    def __init__(self, pool):
        self.pool = pool
        self.entries = OrderedDict()

    def acquire(self, prompt):
        result = []
        size = self.pool.block_size
        # Need at least one uncached token to produce final prompt logits.
        for end in range(size, len(prompt), size):
            key = tuple(prompt[:end])
            if key not in self.entries:
                break
            block = self.entries[key]
            self.entries.move_to_end(key)
            self.pool.retain([block])
            result.append(block)
        return result

    def publish(self, prompt, table, computed):
        size = self.pool.block_size
        for end in range(size, min(len(prompt), computed) + 1, size):
            key = tuple(prompt[:end])
            if key not in self.entries:
                block = table[end // size - 1]
                self.pool.retain([block])
                self.entries[key] = block

    def make_room(self, needed):
        # Only entries with no active request owners can release physical capacity.
        for key, block in list(self.entries.items()):
            if self.pool.n_free >= needed:
                break
            if self.pool.refs[block] == 1:
                del self.entries[key]
                self.pool.release([block])
        return self.pool.n_free >= needed

    def clear(self):
        self.pool.release(list(self.entries.values()))
        self.entries.clear()
