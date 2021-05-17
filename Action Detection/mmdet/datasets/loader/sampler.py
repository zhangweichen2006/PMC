from __future__ import division

import math
import torch
import numpy as np

from torch.distributed import get_world_size, get_rank
from torch.utils.data.sampler import Sampler


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, set1_len=99999, set2_len=99999):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        self.min_len = min(set1_len, set2_len)
        self.set1_len = set1_len
        self.set2_len = set2_len
        # for i, size in enumerate(self.group_sizes):
        #     self.num_samples += int(np.ceil(
        #         size / self.samples_per_gpu)) * self.samples_per_gpu
        # print("self.set1_len",self.set1_len,self.set2_len)
        # print("self.min_len", self.min_len)
        self.num_samples = 2 * self.min_len
        # print("self.num_samples", self.num_samples)
    def __iter__(self):
        indices1 = []
        indices2 = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            indice1 = indice[:self.set1_len].copy()
            indice2 = indice[self.set1_len:].copy()
            # assert len(indice) == size
            assert len(indice1) == self.set1_len
            assert len(indice2) == self.set2_len
            # np.random.shuffle(indice)
            np.random.shuffle(indice1)
            np.random.shuffle(indice2)
            # num_extra = int(np.ceil(size / self.samples_per_gpu)
            #                 ) * self.samples_per_gpu - len(indice)
            # # indice = np.concatenate([indice, indice[:num_extra]])
            # indice1 = np.concatenate([indice1, indice1[:num_extra]])
            # indice2 = np.concatenate([indice2, indice2[:num_extra]])
            # indices.append(indice)
            indices1.append(indice1)
            indices2.append(indice2)
        # indices = np.concatenate(indices)
        indices1 = np.concatenate(indices1)
        indices2 = np.concatenate(indices2)
        # indices1 = [indices1[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
        #                 for i in np.random.permutation(
        #                     range(len(indices1) // self.samples_per_gpu))]

        # indices2 = [indices2[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
        #                 for i in np.random.permutation(
        #                     range(len(indices2) // self.samples_per_gpu))]

        # indices1 = np.concatenate(indices1)
        # indices2 = np.concatenate(indices2)

        # print("len(indices1), len(indices2)", len(indices1), len(indices2))

        indices = np.array(list(zip(indices1, indices2))).flatten()
        indices = torch.from_numpy(indices).long()

        # print("len(indices)", len(indices))
        
        # print("self.num_samples", self.num_samples)

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
