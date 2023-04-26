import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import graphlearning as gl

# Load the mnist dataset
labels = gl.datasets.load('fashionmnist', labels_only=True)

# Define the parameter grid
k_values = [10, 100, 70000]
similarities = ['euclidean', 'angular']
kernels = ['gaussian', 'distance']
num_clusters_values = [8, 10, 12]
extra_dims = [0, 1, 4]
methods = ['NgJordanWeiss', 'ShiMalik', 'NgJordanWeiss-accelerated']

# Create combinations of parameters
param_combinations = list(itertools.product(k_values, similarities, kernels, num_clusters_values, extra_dims, methods))

results = []

# Grid search
for params in param_combinations:
    k, similarity, kernel, num_clusters, extra_dim, method = params

    # Create weight matrix
    W = gl.weightmatrix.knn('fashionmnist', k, metric='vae', similarity=similarity, kernel=kernel)

    # Perform spectral clustering
    model = gl.clustering.spectral(W, num_clusters=num_clusters, method=method, extra_dim=extra_dim)

    # Measure time and accuracy
    start_time = time.time()
    pred_labels = model.fit_predict(all_labels=labels)
    elapsed_time = time.time() - start_time
    accuracy = gl.clustering.clustering_accuracy(pred_labels, labels)

    results.append((params, accuracy, elapsed_time))

    print(f'Parameters: {params}, Accuracy: {accuracy:.2f}%, Time: {elapsed_time:.2f}s')

# Find the best combination
best_combination = max(results, key=lambda x: x[1])
print(f'Best combination: {best_combination[0]}, Accuracy: {best_combination[1]:.2f}%, Time: {best_combination[2]:.2f}s')

# Visualize the results
x = np.arange(len(results))
accuracies = [r[1] for r in results]

plt.figure(figsize=(12, 6))
plt.plot(x, accuracies)
plt.xlabel('Parameter Combination Index')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Different Parameter Combinations')
plt.grid()
plt.show()
