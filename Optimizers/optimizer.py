import numpy as np

class Optimizer:
    """Base optimizer class"""
    
    def __init__(self):
        """Initialize optimizer"""
        pass
    
    def update_params(self, layer):
        """Update layer parameters - to be implemented by subclasses"""
        raise NotImplementedError 