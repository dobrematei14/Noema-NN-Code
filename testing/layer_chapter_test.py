import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from Layer.layer_dense import LayerDense

nnfs.init()

def test_layer_dense_spiral_data():
    X, y = spiral_data(100, 3)
    
    dense = LayerDense(2, 3)
    dense.forward(X)
    
    # Expected first 5 outputs (updated to match current implementation)
    expected_outputs = np.array([
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
        [-1.0475188e-04,  1.1395361e-04, -4.7983500e-05],
        [-2.7414842e-04,  3.1729150e-04, -8.6921798e-05],
        [-4.2188365e-04,  5.2666257e-04, -5.5912682e-05],
        [-5.7707680e-04,  7.1401405e-04, -8.9430439e-05]
    ])
    
    # Test first 5 outputs
    actual_outputs = dense.output[:5]
    
    # Print outputs for verification
    print("Actual outputs:")
    print(actual_outputs)
    print("\nExpected outputs:")
    print(expected_outputs)
    
    # Assert that outputs match expected values
    assert np.allclose(actual_outputs, expected_outputs, atol=1e-10), \
        f"Outputs don't match expected values.\nActual:\n{actual_outputs}\nExpected:\n{expected_outputs}"
    
    print("\nTest passed! Layer_Dense produces expected outputs.")

if __name__ == "__main__":
    test_layer_dense_spiral_data()


# $env:PYTHONPATH="." ; python testing/layer_chapter_test.py