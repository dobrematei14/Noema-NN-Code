import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from Layer.layer_dense import LayerDense
from Activation_Functions.act_func import ActivationReLU, ActivationSoftmax

nnfs.init()

def test_layer_dense_relu():
    X, y = spiral_data(100, 3)
    
    dense = LayerDense(2, 3)
    step = ActivationReLU()
    dense.forward(X)
    step.forward(dense.output)
    
    
    # Expected first 5 outputs
    expected_outputs = np.array([
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 1.1395000e-04, 0.0000000e+00],
        [0.0000000e+00, 3.1729000e-04, 0.0000000e+00],
        [0.0000000e+00, 5.2666000e-04, 0.0000000e+00],
        [0.0000000e+00, 7.1401000e-04, 0.0000000e+00]
    ])
    
    # Test first 5 outputs
    actual_outputs = step.output[:5]
    
    # Print outputs for verification
    print("Actual outputs:")
    print(actual_outputs)
    print("\nExpected outputs:")
    print(expected_outputs)
    
    # Assert that outputs match expected values
    assert np.allclose(actual_outputs, expected_outputs, atol=1e-6), f"Outputs don't match expected values.\nActual:\n{actual_outputs}\nExpected:\n{expected_outputs}" # 1e-6 is the tolerance for floating point errors
    
    print("\nTest passed! ReLU activation function produces expected outputs.")
    
    
def test_layer_dense_softmax():
    X, y = spiral_data(100, 3)
    
    dense = LayerDense(2, 3)
    step = ActivationReLU()    
    dense.forward(X)
    step.forward(dense.output)
    
    dense2 = LayerDense(3, 3)
    softmax = ActivationSoftmax()
    dense2.forward(step.output)
    softmax.forward(dense2.output)
    
    
    # Expected first 5 outputs
    expected_outputs = np.array([
        [0.33333334, 0.33333334, 0.33333334],
        [0.33333316, 0.3333332,  0.33333364],
        [0.33333287, 0.3333329,  0.33333418],
        [0.3333326,  0.33333263, 0.33333477],
        [0.33333233, 0.3333324,  0.33333528]
    ])
    
    # Test first 5 outputs
    actual_outputs = softmax.output[:5]
    
    # Print outputs for verification
    print("Actual outputs:")
    print(actual_outputs)
    print("\nExpected outputs:")
    print(expected_outputs)
    
    # Assert that outputs match expected values
    assert np.allclose(actual_outputs, expected_outputs, atol=1e-6), f"Outputs don't match expected values.\nActual:\n{actual_outputs}\nExpected:\n{expected_outputs}" # 1e-6 is the tolerance for floating point errors
    
    print("\nTest passed! Softmax activation function produces expected outputs.")

if __name__ == "__main__":
    test_layer_dense_relu()
    test_layer_dense_softmax()


# $env:PYTHONPATH="." ; python.exe .\testing\act_func_chapter_test.py