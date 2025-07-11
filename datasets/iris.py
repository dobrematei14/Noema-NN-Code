import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def load_iris():
    path = kagglehub.dataset_download("uciml/iris")
    df = pd.read_csv(os.path.join(path, "iris.csv"))
    
    # Drop the first column (ID)
    df = df.drop(df.columns[0], axis=1)
    
    # Print to see the structure
    print("Dataset columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)
    print("First few rows:\n", df.head())
    
    # Extract features (columns 0:3) and labels (column 4)
    X = df.iloc[:, 0:4].values
    
    # Extract labels
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y = df.iloc[:, 4].map(label_map).values
    
    return X, y

def visualize_iris_lda():
    """Create LDA visualization showing species separation in 2D."""
    X, y = load_iris()
    
    # Standardize and apply LDA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Plot
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['red', 'blue', 'green']
    
    plt.figure(figsize=(10, 8))
    
    for species_idx, (species, color) in enumerate(zip(species_names, colors)):
        species_mask = y == species_idx
        plt.scatter(X_lda[species_mask, 0], X_lda[species_mask, 1], 
                   alpha=0.7, label=species, color=color, s=60)
    
    plt.xlabel('First Linear Discriminant')
    plt.ylabel('Second Linear Discriminant')
    plt.title('Iris Dataset: LDA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    visualize_iris_lda()
    

# Windows:$env:PYTHONPATH="." ; python.exe .\datasets\iris.py
# Mac/Linux: PYTHONPATH=. python datasets/iris.py