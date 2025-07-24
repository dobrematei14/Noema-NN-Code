import numpy as np
from .optimizer import Optimizer

class Optimizer_SGD(Optimizer):
    
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        # Apply learning rate decay if specified
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        
        # Use momentum if specified
        if self.momentum: # allows for a non momentum based optimizer, to see the difference
            
            # Initialize momentum arrays on first call
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            # Calculate momentum-based updates
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        # Standard SGD updates without momentum
        else:
            weight_updates = -self.current_learning_rate * \
                           layer.dweights
            bias_updates = -self.current_learning_rate * \
                         layer.dbiases
        
        # Apply updates to layer parameters
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iterations += 1 