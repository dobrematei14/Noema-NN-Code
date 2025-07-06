import numpy as np

class Neuron:
    """
    Neuron implementation using numpy - multiple inputs, apply weights to each,
    add bias, calculate final output using np.dot for efficient computation.
    """
    
    def __init__(self, weights, bias):
        """
        Args:
            weights (list or np.array): List of weights for each input
            bias (float): Bias term for the neuron
            num_inputs (int): Number of inputs for the neuron
        """
        self.weights = np.array(weights)
        self.bias = bias
        self.num_inputs = len(weights)
    
    def forward(self, inputs):
        """
        Args:
            inputs (list or np.array): List of input values
            
        Returns:
            float: The neuron output
            
        Raises:
            ValueError: If number of inputs doesn't match number of weights
        """
        inputs = np.array(inputs)
        
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Weighted sum using np.dot: np.dot(inputs, weights) + bias
        output = np.dot(inputs, self.weights) + self.bias
        
        return output
    
    def __str__(self):
        """Human-readable neuron"""
        return f"Neuron with {self.num_inputs} inputs, weights={self.weights.tolist()}, bias={self.bias}" 