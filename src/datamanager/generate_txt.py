
import os

tags = ['train', 'val', 'test']

for tag in tags:
    directory = '../../Segmentation_dataset/GTA/image/{}'.format(tag)
    filePatern = '.png'
    outputFile = '../../Segmentation_dataset/GTA/image/{}.txt'.format(tag)
    outputFilter = '../../Segmentation_dataset/GTA/image/'

    path=[]
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filePatern in filename:
                p = os.path.join(root, filename).replace(outputFilter, '')
                path.append(p)

    with open(outputFile, 'w') as f:
        f.write('\n'.join(path))
