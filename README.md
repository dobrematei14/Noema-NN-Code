# Neural Networks from Scratch

Building neural networks from the ground up in Python to understand how they actually work.


## Chapter 1: Neurons

### Overview
Implement a basic neuron from scratch in Python. This is the fundamental building block of neural networks.

### Task
Create a neuron that:

1. Takes multiple inputs from previous layer neurons
2. Applies unique weights to each input
3. Adds a bias term
4. Calculates the final output

### Code Implementation

#### Basic Neuron Formula
```
output = (input₁ × weight₁) + (input₂ × weight₂) + (input₃ × weight₃) + bias
```

#### Example Implementation
```python
# Three inputs from previous layer neurons
inputs = [1.2, 5.1, 2.1]

# Each input has a unique weight
weights = [3.1, 2.1, 8.7]

# Each neuron has a unique bias
bias = 3

# Calculate neuron output
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(f"Neuron output: {output}")  # Expected: 35.7
```

#### Expected Output
```
Neuron output: 35.7
```


## Chapter 2: Layers

### Overview
Implement a layer of neurons that work together. A layer is a collection of neurons that all receive the same inputs but have different weights and biases, producing multiple outputs simultaneously.

### Task
Create a layer that:

1. Contains multiple neurons
2. Processes the same inputs through all neurons in parallel
3. Returns a list of outputs from all neurons
4. Handles edge cases like empty layers or mismatched inputs

### Code Implementation

#### Basic Layer Concept
```
Layer Input: [1, 2, 3, 2.5]

Neuron 1: weights=[0.2, 0.8, -0.5, 1.0], bias=2 → output=4.8
Neuron 2: weights=[0.5, -0.91, 0.26, -0.5], bias=3 → output=1.21  
Neuron 3: weights=[-0.26, -0.27, 0.17, 0.87], bias=0.5 → output=2.385

Layer Output: [4.8, 1.21, 2.385]
```

#### Example Implementation
```python
from Neuron import Neuron
from Layer import Layer

# Create individual neurons with different weights and biases
neuron1 = Neuron([0.2, 0.8, -0.5, 1.0], 2)      # 4 weights, bias=2
neuron2 = Neuron([0.5, -0.91, 0.26, -0.5], 3)   # 4 weights, bias=3
neuron3 = Neuron([-0.26, -0.27, 0.17, 0.87], 0.5) # 4 weights, bias=0.5

# Create a layer containing all neurons
layer = Layer([neuron1, neuron2, neuron3])

# Same inputs go to all neurons in the layer
inputs = [1, 2, 3, 2.5]

# Process inputs through the entire layer
outputs = layer.forward(inputs)
print(f"Layer outputs: {outputs}")  # Expected: [4.8, 1.21, 2.385]

# Layer information
print(f"Number of neurons in layer: {layer.num_neurons}")  # Expected: 3
```

#### Manual Calculation Example
Let's verify neuron1's calculation:
```python
# Neuron 1: weights=[0.2, 0.8, -0.5, 1.0], bias=2
# Inputs: [1, 2, 3, 2.5]

output1 = (1 * 0.2) + (2 * 0.8) + (3 * -0.5) + (2.5 * 1.0) + 2
output1 = 0.2 + 1.6 + (-1.5) + 2.5 + 2
output1 = 4.8
```

#### Expected Output
```
Layer outputs: [4.8, 1.21, 2.385]
Number of neurons in layer: 3
```

#### Error Handling
```python
# All neurons in a layer must expect the same number of inputs
neuron1 = Neuron([0.2, 0.8, -0.5], 2)    # Expects 3 inputs
neuron2 = Neuron([0.5, -0.9], 3)         # Expects 2 inputs - MISMATCH!
layer = Layer([neuron1, neuron2])

inputs = [1, 2, 3]  # 3 inputs provided
outputs = layer.forward(inputs)  # Raises ValueError for neuron2
```
