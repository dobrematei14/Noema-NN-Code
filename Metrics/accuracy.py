import numpy as np

class Accuracy:
    """Base accuracy class"""
    
    def calculate(self, predictions, y_true):
        """Calculate accuracy given predictions and true labels"""
        comparisons = self.compare(predictions, y_true)
        accuracy = np.mean(comparisons)
        return accuracy


class AccuracyCategorical(Accuracy): # extends Accuracy for classification
    """Accuracy for classification"""
    
    def compare(self, predictions, y_true):
        """Compare predictions with true labels for classification"""
        if len(y_true.shape) == 2:  # One-hot encoded
            y_true = np.argmax(y_true, axis=1)
        
        y_pred = np.argmax(predictions, axis=1)
        return y_pred == y_true