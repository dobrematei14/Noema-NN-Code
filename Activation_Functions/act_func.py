import numpy as np
from Losses.loss import Loss_CategoricalCrossentropy

class ActivationStep: # 0 negative, 1 positive activation
    def forward(self, inputs): 
        self.output = np.array([1 if i > 0 else 0 for i in inputs])
        return self.output

class ActivationLinear: # identity activation
    def forward(self, inputs):
        self.output = inputs
        return self.output
    
class ActivationSigmoid: #  (0,1) activation 
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

class ActivationReLU: # 0 negative, linear positive activation

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
    
class ActivationSoftmax: # sum to 1 probability distribution activation
    def forward(self, inputs):
        # Subtract max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        
        return self.dinputs


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combined Softmax activation and Categorical Cross-Entropy loss
    for faster backward step
    """
    
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples