import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def load_pima_diabetes():
    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    
    # Find the CSV file in the downloaded path
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    
    # Print to see the structure
    print("Dataset columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)
    print("First few rows:\n", df.head())
    
    # Extract features (all columns except the last one) and labels (last column)
    X = df.iloc[:, :-1].values
    
    # Extract labels (0 for no diabetes, 1 for diabetes)
    y = df.iloc[:, -1].values
    
    return X, y

def visualize_diabetes_lda():
    """Create LDA histogram showing diabetes/non-diabetes separation."""
    X, y = load_pima_diabetes()
    
    # Standardize and apply LDA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Separate classes
    no_diabetes = X_lda[y == 0].flatten()
    diabetes = X_lda[y == 1].flatten()
    
    # Plot histogram
    plt.figure(figsize=(10, 8))
    
    plt.hist(no_diabetes, bins=30, alpha=0.7, label='No Diabetes', color='blue')
    plt.hist(diabetes, bins=30, alpha=0.7, label='Diabetes', color='red')
    
    plt.xlabel('LDA Component')
    plt.ylabel('Frequency')
    plt.title('Diabetes Dataset: LDA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    visualize_diabetes_lda()
    
    
# Windows: $env:PYTHONPATH="." ; python.exe .\datasets\diabetes.py
# Mac/Linux: PYTHONPATH=. python datasets/diabetes.py