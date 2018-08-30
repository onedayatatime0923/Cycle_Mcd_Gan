

from dataset.concat_dataset import ConcatDataset
from dataset.cycle_dataset import CycleMcdDataset
from dataset.gta_dataset import GTADataSet
from dataset.city_dataset import CityDataSet
from dataset.source_target_dataset import SourceTargetDataset

assert CycleMcdDataset and SourceTargetDataset

def createDataset(datasetList, transform, outputFile):
    dataset = []
    for d in datasetList:
        datasetName, split = d.split('_')
        dataset.append( getDataset(datasetName, split, transform, outputFile))

    return ConcatDataset(dataset)

def getDataset(datasetName, split, transform, outputFile):
    assert datasetName in ["gta", "city"]

    name2class = {
        "gta": GTADataSet,
        "city": CityDataSet,
    }

    name2root = {  ## Fill the directory over images folder. put train.txt, val.txt in this folder
        "gta": "../Segmentation_dataset/GTA/",
        "city": "../Segmentation_dataset/Cityscapes/",
    }
    dataClass = name2class[datasetName]
    dataroot = name2root[datasetName]

    return dataClass(root=dataroot, split=split, transform= transform, outputFile=outputFile)
