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
        self.output = np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        for i in range(len(self.output)):
            self.dinputs[i] = self.output[i] * (1 - self.output[i]) * self.dinputs[i]
        return self.dinputs
    


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = Loss_CategoricalCrossentropy()  # Import this at the top
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples