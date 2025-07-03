class Layer:
    """
    Layer implementation - contains multiple neurons,
    processes inputs through all neurons in the layer.
    """
    
    def __init__(self, neurons):
        """
        Args:
            neurons (list): List of Neuron objects in this layer
        """
        self.neurons = neurons
        self.num_neurons = len(neurons) #number of neurons in the layer
    
    def forward(self, inputs):
        """
        Args:
            inputs (list): List of input values
            
        Returns:
            list: The outputs from all neurons in the layer
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        
        return outputs
    
    def __str__(self):
        """Human-readable layer representation"""
        return f"Layer with {self.num_neurons} neurons, neurons={[str(neuron) for neuron in self.neurons]}" 