'''
Created on Feb 2, 2015

@author: Alexandre
'''
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop
dataset = DogsVsCats(
                     RandomCrop(256, 221),
                     start = 0, stop = 19999)
iterator = dataset.iterator(mode = 'batchwise_shuffled_sequential',
                            batch_size=100)
for X, y in iterator:
    print X.shape, y.shape