import numpy as np
import sys
sys.path.append('..')

from nnfs.datasets import spiral_data
from datasets.iris import load_iris
from datasets.diabetes import load_pima_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from Layer import LayerDense
from Activation_Functions import ActivationReLU
from Activation_Functions.act_func import Activation_Softmax_Loss_CategoricalCrossentropy
from Optimizers import Optimizer_Adam
from Metrics.accuracy import AccuracyCategorical

def test_optimizer_spiral_data():
    """Test optimizer with spiral dataset."""
    print("Testing optimizer with spiral dataset...")
    
    X, y = spiral_data(samples=100, classes=3)

    dense1 = LayerDense(2, 64)
    activation1 = ActivationReLU()
    dense2 = LayerDense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)
    accuracy_fn = AccuracyCategorical()

    # Train in loop - reduced epochs to prevent overfitting
    for epoch in range(10001):

        # Forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)
        
        accuracy = accuracy_fn.calculate(loss_activation.output, y)
        
        if not epoch % 200:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')
        
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    
    print(f"Final spiral data test - Accuracy: {accuracy:.3f}, Loss: {loss:.3f}")
    
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    cm = confusion_matrix(y, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("Spiral data test completed!\n")

def test_optimizer_iris_data():
    """Test optimizer with iris dataset using train/test split."""
    print("Testing optimizer with iris dataset...")
    
    X, y = load_iris()
    
    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    dense1 = LayerDense(4, 8)
    activation1 = ActivationReLU()
    dense2 = LayerDense(8, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-7)
    accuracy_fn = AccuracyCategorical()
    
    # Train in loop - reduced epochs to prevent overfitting
    for epoch in range(500):
        
        # Forward pass
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y_train)
        
        accuracy = accuracy_fn.calculate(loss_activation.output, y_train)
        
        if not epoch % 50:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')
        
        # Backward pass
        loss_activation.backward(loss_activation.output, y_train)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    
    # Test on test set
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    test_loss = loss_activation.forward(dense2.output, y_test)
    
    test_accuracy = accuracy_fn.calculate(loss_activation.output, y_test)
    
    print(f'Test accuracy: {test_accuracy:.3f}')
    print(f'Test loss: {test_loss:.3f}')
    
    test_predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, test_predictions)
    print("Test Set Confusion Matrix:")
    print("Rows: Actual classes, Columns: Predicted classes")
    print("Classes: 0=Setosa, 1=Versicolor, 2=Virginica")
    print(cm)
    print(f"Correct predictions: {np.sum(test_predictions == y_test)}/{len(y_test)}")
    print("Iris data test completed!\n")

def test_optimizer_diabetes_data():
    """Test optimizer with diabetes dataset using train/test split."""
    print("Testing optimizer with diabetes dataset...")
    
    X, y = load_pima_diabetes()
    
    # Balance the dataset by undersampling majority class
    diabetes_indices = np.where(y == 1)[0]
    no_diabetes_indices = np.where(y == 0)[0]
    
    # Use the smaller class size (diabetes) as target
    target_size = len(diabetes_indices)
    
    # Randomly sample from majority class
    np.random.seed(42)
    balanced_no_diabetes_indices = np.random.choice(no_diabetes_indices, target_size, replace=False)
    
    # Combine balanced indices
    balanced_indices = np.concatenate([diabetes_indices, balanced_no_diabetes_indices])
    
    # Create balanced dataset
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print(f"Original dataset: {len(y)} samples")
    print(f"Balanced dataset: {len(y_balanced)} samples (50% each class)")
    
    # Split balanced data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    dense1 = LayerDense(8, 8)
    activation1 = ActivationReLU()
    dense2 = LayerDense(8, 2)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-6)
    accuracy_fn = AccuracyCategorical()
    
    # Train in loop - fewer epochs to prevent overfitting
    for epoch in range(501):
        
        # Forward pass
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y_train)
        
        accuracy = accuracy_fn.calculate(loss_activation.output, y_train)
        
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')
        
        # Backward pass
        loss_activation.backward(loss_activation.output, y_train)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    
    # Test on test set
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    test_loss = loss_activation.forward(dense2.output, y_test)
    
    test_accuracy = accuracy_fn.calculate(loss_activation.output, y_test)
    
    print(f'Test accuracy: {test_accuracy:.3f}')
    print(f'Test loss: {test_loss:.3f}')
    
    test_predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, test_predictions)
    print("Test Set Confusion Matrix:")
    print("Rows: Actual classes, Columns: Predicted classes")
    print("Classes: 0=No Diabetes, 1=Diabetes")
    print(cm)
    print(f"Correct predictions: {np.sum(test_predictions == y_test)}/{len(y_test)}")
    print("Diabetes data test completed!\n")

if __name__ == "__main__":
    test_optimizer_spiral_data()
    test_optimizer_iris_data()
    test_optimizer_diabetes_data() 