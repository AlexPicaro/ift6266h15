from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, Softmax, Tanh
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import MethodCost
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

batchSize = 20
cropSize = 200

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
h0 = ConvRectifiedLinear(output_channels = 40, 
                         kernel_shape = [7,7], 
                         pool_shape = [2,2], 
                         pool_stride = [2,2], 
                         layer_name = "h0", 
                         irange = 0.1, 
                         border_mode = "full")
 
h1 = ConvRectifiedLinear(output_channels = 40, 
                         kernel_shape = [7,7], 
                         pool_shape = [2,2], 
                         pool_stride = [2,2], 
                         layer_name = "h1", 
                         irange = 0.1, 
                         border_mode = "full")

h2 = Tanh(dim = 200,
          layer_name = "h2",
          irange = 0.1)

h3 = Tanh(dim = 200,
          layer_name = "h3",
          irange = 0.1)
 
y = Softmax(n_classes = 2,
            layer_name = "y",
            irange = 0.1)

inputSpace = Conv2DSpace(shape = [cropSize,cropSize],
                         num_channels = 3)
  
model = MLP(layers = [h0, h1, h2, h3, y], 
            batch_size = batchSize, 
            input_space = inputSpace)
 
algorithm = SGD(learning_rate = 0.01, 
                cost = MethodCost("cost_from_X"), 
                batch_size = batchSize, 
                monitoring_batch_size = batchSize,
                monitoring_dataset = {'train': train,
                                      'valid':valid}, 
                monitor_iteration_mode = "even_batchwise_shuffled_sequential", 
                termination_criterion = EpochCounter(max_epochs = 200),
                learning_rule = Momentum(init_momentum = 0.99),
                train_iteration_mode = "even_batchwise_shuffled_sequential")
     
train = Train(dataset = train, 
              model = model, 
              algorithm = algorithm, 
              save_path = "ConvNet4.pkl", 
              save_freq = 1, 
              extensions = [
                            MonitorBasedSaveBest(channel_name = "valid_y_misclass",
                                                 save_path = "ConvNet4_best.pkl"),
			    MomentumAdjustor(final_momentum = 0,
                                             start = 0,
					     saturate = 100)
                            ])
     
print("Starting training session")

train.main_loop()

print("Done!")
