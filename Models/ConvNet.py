'''
Created on Feb 13, 2015

@author: Alexandre
'''

from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop
from pylearn2.models.mlp import ConvRectifiedLinear, Softmax, MLP
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import MethodCost
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

batchSize = 20
cropSize = 150

#Create datasets
train = DogsVsCats(
                     RandomCrop(256, cropSize),
                     start = 0,
                     stop = 19999
                     )
valid = DogsVsCats(
                    RandomCrop(256, cropSize),
                    start = 20000,
                    stop = 22500
                   )

#Instantiate layers
h0 = ConvRectifiedLinear(output_channels = 32, 
                         kernel_shape = [5,5], 
                         pool_shape = [2,2], 
                         pool_stride = [2,2], 
                         layer_name = "h0", 
                         irange = 0.1, 
                         border_mode = "full")
 
h1 = ConvRectifiedLinear(output_channels = 32, 
                         kernel_shape = [5,5], 
                         pool_shape = [2,2], 
                         pool_stride = [2,2], 
                         layer_name = "h1", 
                         irange = 0.1, 
                         border_mode = "full")
 
y = Softmax(n_classes = 2,
            layer_name = "y",
            irange = 0.1)

inputSpace = Conv2DSpace(shape = [cropSize,cropSize],
                         num_channels = 3)
  
model = MLP(layers = [h0, h1, y], 
            batch_size = batchSize, 
            input_space = inputSpace)
 
algorithm = SGD(learning_rate = 1e-3, 
                cost = MethodCost("cost_from_X"), 
                batch_size = batchSize, 
                monitoring_batch_size = batchSize,
                monitoring_dataset = {'train': train,
                                      'valid':valid}, 
                monitor_iteration_mode = "even_batchwise_shuffled_sequential", 
                termination_criterion = EpochCounter(max_epochs = 100), 
                learning_rule = Momentum(init_momentum = 0.0),
                train_iteration_mode = "even_batchwise_shuffled_sequential")
     
train = Train(dataset = train, 
              model = model, 
              algorithm = algorithm, 
              save_path = "2_layer_conv.pkl", 
              save_freq = 1, 
              extensions = [
                            MonitorBasedSaveBest(channel_name = "valid_y_misclass",
                                                 save_path = "2_layer_conv_best.pkl")
                            ])
     
print("Starting training session")

train.main_loop()

print("Done!")