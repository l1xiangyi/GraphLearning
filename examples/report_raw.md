for my optimization final paper, please help me find typo and broken english. do not count mathmatical symbols as typo

# Introduction

Clustering has been an established unsupervised learning method to group similar data with no requirement on labeling the training data. In most scienctific fields that require dealing with empirical data, clustering is one of the most used exploratory data analysis techniques. Since [1] K-means clustering has been widely used. K-means algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-square. However, [2] has shown traditional clustering algorithms have some fundamental limitations such as inability of not clustering non-convex and high-dimensional data well. Thee spectral clustering algorithm, especially the one proposed by [3] is not only easy to implement but also outperforms K-means and other clustering algorithm shown by a comparison of different clustering algorithms from [4]. The spectral clustering algorithm considers good clusters as graphs that maximizes in-cluster edges and minimize inter-cluster cuts. It applies clustering to a projection of the normalized Laplacian. Below is a visualization on a toy dataset: 

# Spectral Clustering

Given a dataset 𝑋 = {𝑥1 , ..., 𝑥𝑛 }, assume the similarity between 𝑥𝑖 and 𝑥𝑗 is 𝑠𝑖𝑗 ≥ 0. The idea of spec- tral clustering is to divide the data points into different clusters so that the points within the same cluster are similar, while the points in different clusters are dissimilar to each other. Here we use sim- ilarity graph 𝐺 = (𝑉 , 𝐸) to represent the dataset.

## Graph Cut 
Let 𝐺 = (𝑉 , 𝐸) be the similarity graph, where 𝑉 = {𝑣1, ..., 𝑣𝑛} and each vertex 𝑣𝑖 denotes a data point. We assume the graph is weighted and each edge between 𝑣𝑖 and 𝑣𝑗 carries a non-negative weight 𝑤𝑖𝑗. en the weighted adjacency matrix is constructed as

The degree matrix of G is defined as:

To cluster the graph into several clusters, we want to minimize the weight of edges between different clusters. In a k-cluster problem, the minicut approach is to choose a partition 𝐴1 , ..., 𝐴𝑘 which mini- mizes

and the optimization problem can be written as

Although mincut is a relatively easy problem to solve, it oen does not lead to satisfactory partitions, for the reason that in many cases, it simply seperates one individual vertex from the rest of the graph. To solve this problem, we want to constrain the size of the clusters so that they are “reasonably large”. Normalized cut (Ncut) is one of the most common objective functions used to solve the problem.

## Normalized cut for k-cluster problem
Normalized cut (Ncut) provides balanced graph partition by constraining the size of the clusters to be similar. Let vol(𝐴𝑖) = ∑𝑖∈𝐴 𝑑𝑖 denotes the size of edges in cluster 𝐴𝑖. Normalized cut for k clusters is defined as:

where 𝐴 stands for the complement of cluster 𝐴𝑖

Then the optimization problem constructed by normalized cut is

e minimum of ∑𝑘 􏰅 1 􏰆 is achieved if all vol(𝐴𝑖) coincide. However, solving this problem is NP 𝑖=1 vol(𝐴)
hard. us we use spectral clustering to solve the relaxed version of the problem.
e objective function can be rewrien using graph Laplacian. Define the cluster inidcator vector h𝑗 = 􏰇h1,𝑗,...,h𝑛,𝑗􏰈𝑇 as