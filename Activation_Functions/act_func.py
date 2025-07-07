import numpy as np

class ActivationStep:
    def forward(self, inputs): 
        self.output = np.array([1 if i > 0 else 0 for i in inputs])
        return self.output

class ActivationLinear:
    def forward(self, inputs):
        self.output = inputs
        return self.output
    
class ActivationSigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output
    
class ActivationSoftmax:
    def forward(self, inputs):
        self.output = np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)
        return self.output
    

