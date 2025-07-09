import numpy as np

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
        self.output = np.maximum(0, inputs)
        return self.output
    
class ActivationSoftmax: # sum to 1 probability distribution activation
    def forward(self, inputs):
        self.output = np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)
        return self.output
    

