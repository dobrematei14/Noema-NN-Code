# Neural Networks from Scratch

Building neural networks from the ground up in Python to understand how they actually work.

## Chapter 1: Neurons

### Overview
A neuron is the fundamental building block of neural networks. It takes multiple inputs, applies weights and bias, then produces an output.

### Formula
```
output = (input₁ × weight₁) + (input₂ × weight₂) + ... + bias
```

### Implementation
```python
# Basic neuron calculation
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = sum(i * w for i, w in zip(inputs, weights)) + bias
```

## Chapter 2: Basic Layer Implementation

### Overview
A layer contains multiple neurons that process the same inputs but with different weights and biases.

### Implementation
```python
from Neuron import Neuron
from Layer import Layer

# Create neurons with different weights/biases
neuron1 = Neuron([0.2, 0.8, -0.5, 1.0], 2)
neuron2 = Neuron([0.5, -0.91, 0.26, -0.5], 3)
neuron3 = Neuron([-0.26, -0.27, 0.17, 0.87], 0.5)

# Create layer
layer = Layer([neuron1, neuron2, neuron3])
outputs = layer.forward([1, 2, 3, 2.5])
```

## Chapter 3: Dense Layer Implementation

### Overview
The `Layer_Dense` class provides an efficient implementation using NumPy with:
- Random weight initialization
- Batch processing capability
- Matrix operations for speed

### Implementation
```python
from Layer.layer_dense import Layer_Dense

# Create dense layer: 4 inputs → 3 neurons
dense_layer = Layer_Dense(4, 3)

# Single sample
output = dense_layer.forward([1.0, 2.0, 3.0, 2.5])

# Batch processing
batch_inputs = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0]]
batch_outputs = dense_layer.forward(batch_inputs)
```

### Key Advantages
- **Efficiency**: Matrix operations vs loops
- **Scalability**: Handles large datasets
- **Batch Processing**: Multiple samples simultaneously
- **GPU Compatible**: NumPy operations

## Chapter 4: Activation Functions

### Overview
Activation functions add non-linearity to neural networks, enabling them to learn complex patterns.

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
Outputs the input if positive, zero otherwise. Most popular activation for hidden layers.

### Softmax
```
Softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ
```
Converts outputs to probabilities (sum to 1). Used in final layer for classification.

### Implementation
```python
from Activation_Functions.act_func import ActivationReLU, ActivationSoftmax
from Layer.layer_dense import Layer_Dense

# Create layers and activations
dense1 = Layer_Dense(4, 5)
relu = ActivationReLU()
dense2 = Layer_Dense(5, 3)
softmax = ActivationSoftmax()

# Forward pass
inputs = [[1.0, 2.0, 3.0, 2.5]]
dense1.forward(inputs)
relu.forward(dense1.output)
dense2.forward(relu.output)
softmax.forward(dense2.output)

print("Probabilities:", softmax.output)
```
