import numpy as np

class Loss: # base class
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) # calculate loss for each sample

        data_loss = np.mean(sample_losses)
        return data_loss # returns the average loss across all samples
    
    
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #clip to avoid log(0), inf or undefined

        if len(y_true.shape) == 1:  # for integer labels like '2'
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # for one-hot encoded labels like [0, 0, 1]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        return -np.log(correct_confidences)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        # Convert y_true to one-hot encoding if it's not already
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((samples, labels))
            y_true_one_hot[np.arange(samples), y_true] = 1
        else:
            y_true_one_hot = y_true
        
        self.dinputs = -y_true_one_hot / dvalues
        self.dinputs /= samples


class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_pred_clipped.shape) == 2:
            y_pred_clipped = y_pred_clipped.flatten()
        
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        
        return sample_losses
    