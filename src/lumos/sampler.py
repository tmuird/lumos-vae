"""Batching for stores whose spectra do not all have the same number of frames.

The encoder pools over the time axis, so every sample in a batch has to share
one window length: a mixed batch would need padding, and the pooling would then
average over the padding. Grouping samples by how many frames they actually have
keeps each batch homogeneous and leaves the model untouched.

A store holding a 16-frame photobleaching series alongside single-frame
acquisitions is therefore two groups, and a batch is drawn from one of them.
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

        # Exact grouping fragments badly when lengths are near-continuous: on
        # glasgow it gives 31 groups, most of them a single spot. Instead walk
        # the distinct lengths from longest to shortest, carrying leftovers down
        # into the next length. A bucket runs at its shortest member's length, so
        # every spot in it has the frames the window needs.
        #
        # The carry is flushed rather than merged when the next length is less
        # than half the current one. Without that, a store mixing 16-frame series
        # with single frames would train some 16-frame spots at T=1, which is the
        # window where the Raman and fluorescence split is unidentifiable.
        by_length = defaultdict(list)
        for i, n in enumerate(self.lengths):
            by_length[int(n)].append(i)

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

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for idx in self.groups.values():
            order = rng.permutation(idx) if self.shuffle else idx
            for start in range(0, len(order), self.batch_size):
                batches.append(order[start:start + self.batch_size].tolist())
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        yield from self._batches()
        self.epoch += 1

    def __len__(self):
        return sum(
            -(-len(idx) // self.batch_size) for idx in self.groups.values()
        )

    def describe(self):
        return ", ".join(
            f"{n} frames x{len(idx)}" for n, idx in self.groups.items()
        )
