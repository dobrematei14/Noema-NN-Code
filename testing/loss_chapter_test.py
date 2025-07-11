import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Layer.layer_dense import LayerDense
from Activation_Functions.act_func import ActivationReLU, ActivationSoftmax, ActivationSigmoid
from Losses.loss import Loss_CategoricalCrossentropy, Loss_BinaryCrossentropy

from datasets.iris import load_iris
from datasets.diabetes import load_pima_diabetes



nnfs.init()

def test_categorical_crossentropy_loss():
    print("Testing categorical crossentropy loss...")
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
    
    print("\nTest passed! Categorical crossentropy loss calculation is correct.")
    print(f"Expected loss: {expected_loss}")
    print(f"Actual loss: {loss}")
    


def test_iris_classification():
    print("\nTesting iris classification...")
    X, y = load_iris()
    
    dense1 = LayerDense(4, 8)
    activation1 = ActivationReLU()
    
    dense2 = LayerDense(8, 3)
    activation2 = ActivationSoftmax() 
    loss_function = Loss_CategoricalCrossentropy()
    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output, y)
    
    
    print(f"\nIris classification loss: {loss}")
    


def test_diabetes_classification():
    print("\nTesting diabetes classification...")
    X, y = load_pima_diabetes()
    
    dense1 = LayerDense(8, 16)
    activation1 = ActivationReLU()
    
    dense2 = LayerDense(16, 1)
    activation2 = ActivationSigmoid()
    loss_function = Loss_BinaryCrossentropy()  # Use binary crossentropy for binary classification
    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output, y)
    
    
    print(f"\nDiabetes classification loss: {loss}")
    


if __name__ == "__main__":
    test_categorical_crossentropy_loss()
    test_iris_classification()
    test_diabetes_classification()


# PYTHONPATH=. python testing/loss_chapter_test.py
# Windows: $env:PYTHONPATH="." ; python.exe .\testing\loss_chapter_test.py
# Mac/Linux: PYTHONPATH=. python testing/loss_chapter_test.py


