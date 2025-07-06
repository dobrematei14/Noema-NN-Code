import numpy as np

class Layer_Dense:
    """
    Layer implementation using random initialization - compatible with batch processing
    as well as single sample processing.
    """
    
    def __init__(self, num_inputs, num_neurons):
        """
        Args:
            num_inputs (int): Number of inputs for the layer
            num_neurons (int): Number of neurons in the layer
        """
        # follow book initialization to match the results
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons) # size (num_inputs, num_neurons), instead of (num_neurons, num_inputs) to avoid transpose
        self.biases = np.zeros((1, num_neurons))
    
    def forward(self, inputs):
        """
        Args:
            inputs (list): List of input values
            
        Returns:
            list: The outputs from all neurons in the layer
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
 