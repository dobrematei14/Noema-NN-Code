class Neuron:
    """
    Neuron implementation - multiple inputs, apply weights to each,
    add bias, calculate final output.
    """
    
    def __init__(self, weights, bias):
        """
        Args:
            weights (list): List of weights for each input
            bias (float): Bias term for the neuron
            num_inputs (int): Number of inputs to the neuron
        """
        self.weights = weights
        self.bias = bias
        self.num_inputs = len(weights)
    
    def forward(self, inputs):
        """
        Args:
            inputs (list): List of input values
            
        Returns:
            float: The neuron output
            
        Raises:
            ValueError: If number of inputs doesn't match number of weights
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        #weighted sum: sum(input_i * weight_i) + bias
        output = 0
        for i in range(self.num_inputs): # or use a lambda function/zip
            output += inputs[i] * self.weights[i]
        output += self.bias
        
        return output
    
    
    def __str__(self):
        """Human-readable neuron"""
        return f"Neuron with {self.num_inputs} inputs, weights={self.weights}, bias={self.bias}"


