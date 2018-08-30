import os, shutil
import scipy.io
assert shutil

directory = '../../Segmentation_dataset/GTA/image'
splitFile = '../../Segmentation_dataset/GTA/mapping/split.mat'

split = scipy.io.loadmat(splitFile)
trainIds = list(split['trainIds'].reshape(-1))
valIds = list(split['valIds'].reshape(-1))
testIds = list(split['testIds'].reshape(-1))

trainDir='{}/train'.format(directory)
valDir='{}/val'.format(directory)
testDir='{}/test'.format(directory)
if not os.path.exists(trainDir): os.mkdir(trainDir)
if not os.path.exists(valDir): os.mkdir(valDir)
if not os.path.exists(testDir): os.mkdir(testDir)

i = 0

for filename in os.listdir(directory):
    if filename == 'train' or filename == 'val' or filename == 'test':
        pass
    elif int(filename.split('.')[0]) in trainIds:
        originalPath = os.path.join(directory, filename)
        newPath = os.path.join(trainDir, filename)
        i += 1
        os.rename(originalPath, newPath)
        print(i)
    elif int(filename.split('.')[0]) in valIds:
        originalPath = os.path.join(directory, filename)
        newPath = os.path.join(valDir, filename)
        i += 1
        os.rename(originalPath, newPath)
        print(i)
    elif int(filename.split('.')[0]) in testIds:
        originalPath = os.path.join(directory, filename)
        newPath = os.path.join(testDir, filename)
        i += 1
        os.rename(originalPath, newPath)
        print(i)
    else:
        raise NotImplementedError
