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