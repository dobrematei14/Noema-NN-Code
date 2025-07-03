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
Coming soon...

## Chapter 3: Activation Functions
Coming soon...

## Chapter 4: Optimization
Coming soon...

## Chapter 5: Backpropagation
Coming soon...