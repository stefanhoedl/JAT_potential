# JAT potential

A deep learning architecture to predict the potential energy and forces of organic molecules and ionic liquids. 
The architecture uses message passing neural networks (MPNN^1) with an attentional update function (linear dynamic attention, GATv2^2) by adapting Graph Attention Networks (GAT^3) to the domain of computational chemistry. The name JAT (JraphAttentionNetworks) derives from adapting Graph Attention Networks in JAX and builds upon the Jraph library. 

The code for the JAT model was developed during the master's thesis at TU Vienna with the department of Theoretical Materials Chemistry under supervision of Dr. Jesús Carrete Montaña. 

In this thesis, I've
- built a deep learning architecture by adapting a state-of-the-art DL approach (Graph Attention Networks) to the domain of computational chemistry
- adapted the NeuralIL codebase (which uses conventional and computationally expensive fingerprint features and a shallow neural network)
- performed an extensive literature review surveying the state-of-the-art in multiple fields (computational chemistry, graph-based/geometric deep learning, attention & Transformer-based architectures) to extract the most promising approaches
- optimized, debugged and trained the architecture on a small dataset of ionic liquids
- scaled the architecture to the very large ANI-1 dataset
- while optimizing for efficiency of the architecture and achieving a 4x speedup over the supervisors' baseline with comparable accuracy.


## Overview
Domain:        Computational chemistry

Goal:           Predict the energy of molecules

Task:           Learn the potential energy surface (PES): 
atomic positions → energy mapping

high-dimensional task (3N - 6 degrees of freedom for N atoms)

Problem:     "Exponential barrier" of electronic contribution 

Solution:        Approximate the Schrödinger Equation efficient Deep Neural Networks

## Approach & Architecture
- Adapt NeuralIL implementation [1]
- replace spherical Bessel descriptors with a Message Passing Neural Network [2]
- using Graph Attention Networks [3]
- on a sparse molecular graph

JAT potential: Jraph Attention Network
to predict the potential energy and atomic forces of molecules
Extends the NeuralIL architecture and implementation
adapts Graph Attention Networks (GAT) to replace fingerprint features
within the message passing neural networks (MPNN) framework

performs multiple message passing steps by
- generating a sparse molecular graph
- using dynamic linear attention (GATv2) as the message function
    (masked multi-head self-attention)
- weighted sum of features + skip connection as the update function
- pyramidal regression head as readout function


## JAT architecture
![Visualization of the entire JAT architecture](figs/jatOnlyModel.png)
Visualization of the entire JAT architecture. Using the positions and species (types) as input, the graph generator (Fig. \ref{fig:vis_graphGen}) generates a sparse edge list, with which T message passing steps (here 4) using an attentional update function (Fig. \ref{fig:vis_attention}) are performed. The features $h^0$ at $t=0$ are obtained from the type embedding layer, for $t=1 - T$ the output features of the last message passing steps serve as input for the next JAT layer. (Fig. \ref{fig:vis_jatLayer}) \\
The readout head (here visualized truncated) transform the features $h^T$ using projection, nonlinearity and normalization layers into the energy contribution for each individual atom. The JAT model's prediction for the potential energy $E_{pot}$ is obtained as the sum over all atoms' contributions, while the forces $\textbf{f}$ are obtained as the model's derivative with respect to the atomic positions

## Graph Generator
![Visualization of the Graph Generator](figs/vis_graphGen.png)
 of the graph generator component of the JAT architecture. Using the Cartesian coordinates (positions) of all atoms, the pairwise distance matrix is calculated using the Euclidean $L^2$ norm under consideration of periodic boundary conditions. The distances are filtered using the graph cut parameter to only include pairs within close proximity, which are connected by an edge. This generated molecular graph is represented as an *edge list*
  in a sparse format using three arrays, respectively the triplets of sender, receiver and distance. Since the number of edges depends on the positions, the edge list is padded with masked edges up to a static maximum.

## JAT Layer
![Visualization of a JAT layer](figs/vis_JatLayer.png)
Visualization of a single JAT layer, which performs a single round of message passing to update the node feature vectors. The features $\textbf{h}^{t}$ at step $t$ are projected into senders and receivers, and for every pair in the edge list a weight $\alpha$ is calculated using the attention mechanism. The messages are calculated as the element-wise multiplication of the sender features and attention weights $\alpha$. These messages are aggregated to each receiver using a segment sum and transformed with a nonlinearity, skip connection and layer normalization to obtain the updated node features $\textbf{h}^{t+1}$. These are fed into the next JAT layer to repeat the message passing procedure for multiple rounds.

## Linear dynamic attention
![Visualization of the attention mechanism](figs/vis_attention.png)
Visualization of the attention mechanism of the JAT architecture. For every $\mathrm{edge}_{ij}$ in the edge list, the features of $\mathrm{sender}_{i}$, $\mathrm{receiver}_{j}$ and $d_{ij}$ are *lifted* and with a projection parametrized by $a^T$ transformed into $e_{ij}$. These weights are normalized over all received messages with a segment softmax function to obtain $\alpha_{ij}$. }

## Going fast
- replace computationally expensive descriptors
- implemented in JAX leveraging a GPU
- just-in-time (JIT) compilation
- sparse graph for sparse self-attention 
  O(N2) → O(N * max(Nneighbours) for fully connected graph
- mask for edges & atoms for JIT static shape 


## Features
- invariant to rotation, translation and permutation
- additive ansatz: sum of atomic contributions
- trains on energy or forces
- locality for MD parallelization
- smooth nonlinearity (swish-1) and loss function (log cosh)




## Atomic Forces: Evaluation of the atomic forces via JVP
Using jax.vjp avoids calculating and storing the entire Jacobian when evaluating the atomic forces, which drastically accelerates the model. 
![Visualization of the entire JAT architecture](figs/vis_jatArchitecture.png)


## References
[1] Hadrián Montes-Campos, Jesús Carrete, Sebastian Bichelmaier, Luis M Varela, and Georg KH Madsen. A differentiable neural-network force field for ionic liquids. Journal of chemical information and modeling, 62(1):88–101, 2021

[2] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. PMLR, 2017. 

[3] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. stat, 1050:20, 2017.

