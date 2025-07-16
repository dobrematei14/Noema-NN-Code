import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Layer.layer_dense import LayerDense
from Activation_Functions.act_func import ActivationReLU, ActivationSoftmax
from Losses.loss import Loss_CategoricalCrossentropy

def test_random_optimization():
    """Test random weight optimization approach."""
    
    from nnfs.datasets import vertical_data
    X, y = vertical_data(samples=100, classes=3)
    
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_function = Loss_CategoricalCrossentropy()
    
    lowest_loss = 9999999
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()
    best_accuracy = 0
    improvements_found = 0
    
    print("Random optimization: generating completely random weights each iteration")
    
    for iteration in range(10000):
        dense1.weights = 0.05 * np.random.randn(2, 3)
        dense1.biases = 0.05 * np.random.randn(1, 3)
        dense2.weights = 0.05 * np.random.randn(3, 3)
        dense2.biases = 0.05 * np.random.randn(1, 3)
        
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        
        loss = loss_function.calculate(activation2.output, y)
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)
        
        if loss < lowest_loss:
            improvements_found += 1
            if improvements_found <= 5:
                print(f'Improvement {improvements_found} at iteration {iteration}: Loss={loss:.6f}, Accuracy={accuracy:.3f}')
            
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
            best_accuracy = accuracy
    
    # Restore best weights for final evaluation
    dense1.weights = best_dense1_weights.copy()
    dense1.biases = best_dense1_biases.copy()
    dense2.weights = best_dense2_weights.copy()
    dense2.biases = best_dense2_biases.copy()
    
    # Run forward pass with best weights to get final results
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    final_loss = loss_function.calculate(activation2.output, y)
    final_predictions = np.argmax(activation2.output, axis=1)
    final_accuracy = np.mean(final_predictions == y)
    
    print(f"\nResults: {improvements_found} improvements, final loss: {final_loss:.6f}, accuracy: {final_accuracy:.3f}")
    print(f"Random baseline: 0.333\n")


def test_random_adjustment_optimization():
    """Test random weight adjustment optimization approach."""
    
    from nnfs.datasets import vertical_data
    X, y = vertical_data(samples=100, classes=3)
    
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_function = Loss_CategoricalCrossentropy()
    
    lowest_loss = 9999999
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()
    best_accuracy = 0
    
    improvements_found = 0
    rejected_changes = 0
    
    print("Random adjustment optimization: making small random changes to weights")
    
    for iteration in range(10000):
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)
        
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        
        loss = loss_function.calculate(activation2.output, y)
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)
        
        if loss < lowest_loss:
            improvements_found += 1
            if improvements_found <= 5:
                print(f'Improvement {improvements_found} at iteration {iteration}: Loss={loss:.6f}, Accuracy={accuracy:.3f}')
            
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
            best_accuracy = accuracy
        else:
            rejected_changes += 1
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()
    
    print(f"\nResults: {improvements_found} improvements, {rejected_changes} rejections")
    print(f"Success rate: {(improvements_found/10000)*100:.1f}%")
    print(f"Final loss: {lowest_loss:.6f}, accuracy: {best_accuracy:.3f}")
    print(f"Random baseline: 0.333\n")


if __name__ == "__main__":
    test_random_optimization()
    test_random_adjustment_optimization()


# python -m testing.optimization_chapter_loss