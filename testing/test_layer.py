from Neuron import Neuron
from Layer import Layer

def test_layer_forward_single():
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2
    neuron = Neuron(weights, bias)
    layer = Layer([neuron])
    
    inputs = [1, 2, 3, 2.5]
    expected_output = [4.8]
    actual_output = layer.forward(inputs)
    
    assert len(actual_output) == len(expected_output)
    assert abs(actual_output[0] - expected_output[0]) < 0.001 # account for floating point errors

def test_layer_forward_multiple():
    neuron1 = Neuron([0.2, 0.8, -0.5, 1.0], 2)
    neuron2 = Neuron([0.5, -0.91, 0.26, -0.5], 3)
    neuron3 = Neuron([-0.26, -0.27, 0.17, 0.87], 0.5)
    
    layer = Layer([neuron1, neuron2, neuron3])
    
    inputs = [1, 2, 3, 2.5]
    expected_outputs = [4.8, 1.21, 2.385]
    actual_outputs = layer.forward(inputs)
    
    assert len(actual_outputs) == len(expected_outputs)
    for i in range(len(expected_outputs)):
        assert abs(actual_outputs[i] - expected_outputs[i]) < 0.001

def test_layer_str():
    neuron1 = Neuron([0.2, 0.8, -0.5], 2)
    neuron2 = Neuron([0.5, -0.9, 0.26], 3)

    layer = Layer([neuron1, neuron2])
    expected_str = f"Layer with 2 neurons, neurons=['{neuron1}', '{neuron2}']"
    assert str(layer) == expected_str

def test_layer_empty():
    layer = Layer([])
    
    inputs = [1, 2, 3]
    outputs = layer.forward(inputs)
    
    assert outputs == []
    assert layer.num_neurons == 0

def test_layer_invalid_input():
    neuron1 = Neuron([0.2, 0.8, -0.5], 2)
    neuron2 = Neuron([0.5, -0.9], 3)
    layer = Layer([neuron1, neuron2])
    
    inputs = [1, 2, 3]
    try:
        layer.forward(inputs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected 2 inputs, got 3" in str(e)

if __name__ == "__main__":
    test_layer_forward_single()
    test_layer_forward_multiple()
    test_layer_str()
    test_layer_empty()
    test_layer_invalid_input()
    print("All tests passed!")
