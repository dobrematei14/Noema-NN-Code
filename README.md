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

## Chapter 5: Loss Functions

### Overview
Loss functions measure how far predictions are from actual values. Networks minimize this value during training.

### Categorical Cross Entropy
```
Loss = -log(predicted_probability_of_correct_class)
```
Used for multi-class classification with softmax outputs.

### Mathematical Function

Single sample:
$$L = -\log(p_i)$$

Batch of samples:
$$L = -\frac{1}{N} \sum_{i=1}^{N} \log(p_i)$$

General form:
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij})$$

### Component Breakdown
- **$N$**: Number of samples in batch
- **$C$**: Number of classes
- **$p_i$**: Predicted probability for correct class of sample $i$
- **$p_{ij}$**: Predicted probability that sample $i$ belongs to class $j$
- **$y_{ij}$**: 1 if sample $i$ belongs to class $j$, 0 otherwise
- **$\log$**: Natural logarithm (base e)
- **Negative sign**: Makes loss positive (since log of probabilities is negative)

### Usage
```python
from Losses.loss import Loss_CategoricalCrossentropy

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(predictions, true_labels)
```

## Chapter 6: Optimizers

### Overview
Optimizers update neural network parameters (weights and biases) based on gradients to minimize the loss function. Different optimizers use various strategies to make training more efficient and stable.

### SGD (Stochastic Gradient Descent)
The most basic optimizer that updates parameters in the direction opposite to gradients.

**Basic Update Rule:**
```
weight = weight - learning_rate × gradient
```

**With Momentum:**
```
momentum = β × momentum + learning_rate × gradient
weight = weight - momentum
```

### Adagrad (Adaptive Gradient)
Adapts learning rates for each parameter based on historical gradients. Parameters with large gradients get smaller effective learning rates.

**Update Rule:**
```
cache = cache + gradient²
weight = weight - learning_rate × gradient / (√cache + ε)
```

### RMSprop (Root Mean Square Propagation)
Improves upon Adagrad by using exponential moving average instead of accumulating all gradients, preventing learning rates from becoming too small.

**Update Rule:**
```
cache = ρ × cache + (1-ρ) × gradient²
weight = weight - learning_rate × gradient / (√cache + ε)
```

### Adam (Adaptive Moment Estimation)
Combines momentum with adaptive learning rates and includes bias correction. Generally performs well across various problems.

**Update Rules:**
```
momentum = β₁ × momentum + (1-β₁) × gradient
cache = β₂ × cache + (1-β₂) × gradient²
momentum_corrected = momentum / (1 - β₁ᵗ)
cache_corrected = cache / (1 - β₂ᵗ)
weight = weight - learning_rate × momentum_corrected / (√cache_corrected + ε)
```

### Implementation
```python
from Optimizers.optimizer_sgd import Optimizer_SGD
from Optimizers.optimizer_adam import Optimizer_Adam
from Optimizers.optimizer_rmsprop import Optimizer_RMSprop
from Optimizers.optimizer_adagrad import Optimizer_Adagrad

# Create different optimizers
sgd = Optimizer_SGD(learning_rate=0.1, momentum=0.9, decay=1e-3)
adam = Optimizer_Adam(learning_rate=0.001, decay=1e-4)
rmsprop = Optimizer_RMSprop(learning_rate=0.001, rho=0.9)
adagrad = Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)

# Training loop example
for epoch in range(epochs):
    for batch in batches:
        # Forward pass
        layer1.forward(batch)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        
        # Calculate loss
        loss = loss_function.calculate(activation2.output, targets)
        
        # Backward pass (calculate gradients)
        # ... backpropagation code ...
        
        # Update parameters
        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.post_update_params()
```

### Key Features

- **Learning Rate Decay**: All optimizers support learning rate decay to reduce learning rate over time
- **Adaptive Learning Rates**: Adagrad, RMSprop, and Adam adjust learning rates per parameter
- **Momentum**: SGD and Adam use momentum to accelerate convergence and reduce oscillations
- **Bias Correction**: Adam includes bias correction to handle initialization effects
- **Numerical Stability**: All optimizers include epsilon (ε) to prevent division by zero

### Optimizer Comparison

| Optimizer | Best For | Advantages | Considerations |
|-----------|----------|------------|----------------|
| **SGD** | Simple problems, well-tuned hyperparameters | Simple, reliable, good with momentum | Requires careful learning rate tuning |
| **Adagrad** | Sparse gradients, early training | Adaptive learning rates | Learning rate can become too small |
| **RMSprop** | Non-stationary objectives | Maintains reasonable learning rates | Less common than Adam |
| **Adam** | General purpose, default choice | Combines best of momentum + adaptivity | Can sometimes converge to suboptimal solutions |
