
from dataset.base_dataset import BaseDataset

class ConcatDataset(BaseDataset):
    def __init__(self, dataset):
        index = []
        for datasetIndex, d in enumerate(dataset):
            index.extend([[datasetIndex, i] for i in range(len(d))])
        self.index = index
        self.dataset = dataset

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        _datasetIndex = self.index[i][0]
        _index = self.index[i][1]
        data = self.dataset[_datasetIndex][_index]
        return data
    def name(self):
        return 'ConcatDataSet'
