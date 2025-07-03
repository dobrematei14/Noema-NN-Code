from Neuron import Neuron

def test_neuron_forward():
    weights = [3.1, 2.1, 8.7]
    bias = 3
    neuron = Neuron(weights, bias)
    
    inputs = [1.2, 5.1, 2.1]
    expected_output = 35.7
    actual_output = neuron.forward(inputs)
    
    assert actual_output == expected_output

def test_neuron_str():
    weights = [3.1, 2.1, 8.7]
    bias = 3
    neuron = Neuron(weights, bias)
    expected_str = "Neuron with 3 inputs, weights=[3.1, 2.1, 8.7], bias=3"
    assert str(neuron) == expected_str

def test_neuron_invalid_input_length():
    weights = [3.1, 2.1, 8.7]
    bias = 3
    neuron = Neuron(weights, bias)
    
    inputs = [1.0, 2.0]
    try:
        neuron.forward(inputs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected 3 inputs, got 2" in str(e)

if __name__ == "__main__":
    test_neuron_forward()
    test_neuron_str()
    test_neuron_invalid_input_length()
    print("All tests passed!")