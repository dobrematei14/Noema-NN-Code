import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Layer.layer_dense import LayerDense
from Activation_Functions.act_func import ActivationReLU, ActivationSoftmax
from Losses.loss import Loss_CategoricalCrossentropy
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

def test_layer_backpropagation():
    """Test backpropagation on full layers with batch processing."""
    
    X, y = spiral_data(samples=100, classes=3)
    
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_function = Loss_CategoricalCrossentropy()
    
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    print(f"Before backprop: Loss={loss:.6f}, Accuracy={accuracy:.3f}")
    
    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    print(f"Dense1 gradients: weights={dense1.dweights.shape}, biases={dense1.dbiases.shape}")
    print(f"Dense2 gradients: weights={dense2.dweights.shape}, biases={dense2.dbiases.shape}")
    print()

    print(f"After backprop: Loss={loss:.6f}, Accuracy={accuracy:.3f}")
    
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)
    print()


if __name__ == "__main__":
    test_layer_backpropagation()


# python -m testing.test_backpropagation_chapter9