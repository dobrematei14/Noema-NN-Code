import numpy as np
from .optimizer import Optimizer

class Optimizer_RMSprop(Optimizer):
    
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, 
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        
        # Initialize cache arrays on first call - tracks gradient history per parameter
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache using exponential moving average - unlike AdaGrad which accumulates forever
        # This prevents learning rates from becoming too small over time
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2
        
        # Apply adaptive updates - maintains more consistent learning rates than AdaGrad
        # by "forgetting" older gradients through exponential decay
        layer.weights += -self.current_learning_rate * \
                        layer.dweights / \
                        (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                       layer.dbiases / \
                       (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1 