"""
To avoid padding this is a system to extract subsets of the data such that each sample has the same number of frames
I.e if a dataset is a combination of 16 and 1 frames it will group them to extract batches
"""

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    """Yield batches of indices that all have the same number of valid frames.

    Parameters
    ----------
    lengths : array of int
        Valid frames per sample.
    batch_size : int
        Maximum batch size. A group smaller than this yields one short batch
        rather than being dropped, so rare lengths still train.
    shuffle : bool
        Shuffle within groups and across the batch order each epoch.
    seed : int
        Base seed; the epoch is added so shuffling differs between epochs.
    """

    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths, dtype=int)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        by_length = defaultdict(list)
        for i, n in enumerate(self.lengths):
            by_length[int(n)].append(i)
            # go over indexes and create a dictionary of indexes for each length
        self.groups = defaultdict(list)
        carry = []
        for n in sorted(by_length, reverse=True):
            if carry and n * 2 < int(self.lengths[carry].min()):
                self.groups[int(self.lengths[carry].min())].extend(carry)
                carry = []
            pool = carry + by_length[n]
            full = (len(pool) // batch_size) * batch_size
            if full:
                self.groups[n].extend(pool[:full])
            carry = pool[full:]
        if carry:
            self.groups[int(self.lengths[carry].min())].extend(carry)
        self.groups = {n: np.array(v) for n, v in sorted(self.groups.items())}

    def _group_batches(self, idx, rng):
        order = rng.permutation(idx) if self.shuffle else idx
        return [order[i:i + self.batch_size].tolist()
                for i in range(0, len(order), self.batch_size)]

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = [b for idx in self.groups.values()
                   for b in self._group_batches(idx, rng)]
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        yield from self._batches()
        self.epoch += 1

    def __len__(self):
        return sum(-(-len(idx) // self.batch_size)
                   for idx in self.groups.values())

    def describe(self):
        return ", ".join(
            f"{n} frames x{len(idx)}" for n, idx in self.groups.items()
        )
