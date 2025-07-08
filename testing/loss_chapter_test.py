import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from Layer.layer_dense import LayerDense
from Activation_Functions.act_func import ActivationReLU, ActivationSoftmax
from Losses.loss import Loss_CategoricalCrossentropy

nnfs.init()

def test_categorical_crossentropy_loss():
    X, y = spiral_data(samples=100, classes=3)
    
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_function = Loss_CategoricalCrossentropy()
    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    print("First 5 outputs:")
    print(activation2.output[:5])
    
    loss = loss_function.calculate(activation2.output, y)
    print('loss:', loss)
    
    # Expected loss value (approximately)
    expected_loss = 1.0986104
    
    assert abs(loss - expected_loss) < 0.01, f"Loss doesn't match expected value.\nActual: {loss}\nExpected: {expected_loss}"
    
    print(f"\nTest passed! Categorical crossentropy loss calculation is correct.")
    print(f"Expected loss: {expected_loss}")
    print(f"Actual loss: {loss}")

if __name__ == "__main__":
    test_categorical_crossentropy_loss()


# PYTHONPATH=. python testing/loss_chapter_test.py
