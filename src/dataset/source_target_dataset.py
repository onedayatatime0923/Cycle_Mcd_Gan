
import random
from dataset.base_dataset import BaseDataset

class SourceTargetDataset(BaseDataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __getitem__(self, i):
        return ( self.source[random.randint(0,len(self.source)-1)],
            self.target[random.randint(0,len(self.target)-1)])

    def __len__(self):
        return max(len(self.source), len(self.target))

    def name(self):
        return 'SourceTargetDataset'
