from tsai.all import *
from torch.utils.data import Dataset
from fastai.data.core import DataLoader as FastAIDataLoader
import torch
import torch.nn as nn


class TSDatasetClass(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.LongTensor(y) if not isinstance(y, torch.Tensor) else y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def new_empty(self):
        """Return a new empty dataset of the same shape"""
        if len(self.X) > 0:
            empty_X = torch.zeros((0,) + tuple(self.X.shape[1:]), dtype=self.X.dtype)
            empty_y = torch.zeros((0,), dtype=self.y.dtype)
        else:
            empty_X = torch.zeros((0, 0, 0), dtype=torch.float32)
            empty_y = torch.zeros((0,), dtype=torch.long)

        return TSDatasetClass(empty_X, empty_y)


class TSDataLoaderClass(FastAIDataLoader):
    def __init__(self, X, y, bs=5, shuffle=True):
        dataset = TSDatasetClass(X, y)
        super().__init__(dataset, batch_size=bs, shuffle=shuffle)
        self.bs = bs
        self.shuffle = shuffle
        self.n = len(dataset)
        self.c = len(torch.unique(dataset.y))
        self.dataset = dataset

    def one_batch(self):
        it = iter(self)
        return next(it)

    def new(self, dataset=None, **kwargs):
        """Create new with optional dataset"""
        if dataset is None:
            dataset = self.dataset
        return self.__class__(dataset.X, dataset.y, self.bs, self.shuffle)

    def __iter__(self):
        self.random = torch.Generator()
        self.random.manual_seed(self.rng.randint(0, int(1e6)) if self.shuffle else 0)
        return iter(super().__iter__())
