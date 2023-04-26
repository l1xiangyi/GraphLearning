from .. import graphlearning as gl


"""knn weight matrix
======

General function for constructing knn weight matrices.

Parameters
----------
data : (n,m) numpy array, or string 
    If numpy array, n data points, each of dimension m, if string, then 'mnist', 'fashionmnist', or 'cifar'
k : int
    Number of nearest neighbors to use.
kernel : string (optional), {'uniform','gaussian','singular','distance'}, default='gaussian'
    The choice of kernel in computing the weights between \\(x_i\\) and each of its k 
    nearest neighbors. We let \\(d_k(x_i)\\) denote the distance from \\(x_i\\) to its kth 
    nearest neighbor. The choice 'uniform' corresponds to \\(w_{i,j}=1\\) and constitutes
    an unweighted k nearest neighbor graph, 'gaussian' corresponds to
    \\[ w_{i,j} = \\exp\\left(\\frac{-4\\|x_i - x_j\\|^2}{d_k(x_i)^2} \\right), \\]
    'distance' corresponds to
    \\[ w_{i,j} = \\|x_i - x_j\\|, \\]
    and 'singular' corresponds to 
    \\[ w_{i,j} = \\frac{1}{\\|x_i - x_j\\|}, \\]
    when \\(i\\neq j\\) and \\(w_{i,i}=1\\).
eta : python function handle (optional)
    If provided, this overrides the kernel option and instead uses the weights
    \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{d_k(x_i)^2} \\right), \\]
    where \\(d_k(x_i)\\) is the distance from \\(x_i\\) to its kth nearest neighbor.
symmetrize : bool (optional), default=True, except when kernel='singular'
    Whether or not to symmetrize the weight matrix before returning. Symmetrization is 
    performed by returning \\( (W + W^T)/2 \\), except for when kernel='distance, in 
    which case the symmetrized edge weights are the true distances. Default for symmetrization
    is True, unless the kernel is 'singular', in which case it is False.
metric : string (optional), default='raw'
    Metric identifier if data is a string (i.e., a dataset).
similarity : {'euclidean','angular','manhattan','hamming','dot'} (optional), default='euclidean'
    Smilarity for nearest neighbor search.
knn_data : tuple (optional), default=None
    If desired, the user can provide knn_data = (knn_ind, knn_dist), the output of a knnsearch,
    in order to bypass the knnsearch step, which can be slow for large datasets.
"""

"""
Spectral clustering
        ===================

        Implements several methods for spectral clustering, including Shi-Malik and Ng-Jordan-Weiss. See
        the tutorial paper [1] for details.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object
            Weight matrix representing the graph.
        num_clusters : int
            Number of desired clusters.
        method : {'combinatorial', 'ShiMalik', 'NgJordanWeiss'} (optional), default='NgJordanWeiss'
            Spectral clustering method.
        extra_dim : int (optional), default=0
            Extra dimensions to include in spectral embedding.
"""


# W = gl.weightmatrix.knn('mnist', 10, metric='vae')
W = gl.weightmatrix.knn('mnist', 100, metric='vae')
print(W.shape) # (70000, 70000)
labels = gl.datasets.load('mnist', labels_only=True)
print(labels.shape) # (70000,)

model = gl.clustering.spectral(W, num_clusters=10, extra_dim=0)
pred_labels = model.fit_predict(all_labels=labels)

accuracy = gl.clustering.clustering_accuracy(pred_labels,labels)
print('Clustering Accuracy: %.2f%%'%accuracy)


