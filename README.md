# JAT_potential
JAT (JraphAttentionNetworks), a deep learning architecture to predict the potential energy and forces of molecules. The architecture uses message passing neural networks with an attentional update function (linear dynamic attention, GATv2) by adapting Graph Attention Networks (GAT) to the domain of computational chemistry.

The code for the JAT model was developed during the master's thesis at TU Vienna under supervision of the department of Theoretical Materials Chemistry. 

In my thesis, I've
- built a deep learning architecture by adapting a state-of-the-art DL approach (Graph Attention Networks) to the domain of computational chemistry
- adapted the codebase of my supervisor (which uses conventional and computationally expensive fingerprint features and shallow neural network)
- performed an extensive literature review surveying the state-of-the-art in multiple fields (computational chemistry, graph-based/geometric deep learning, attention & Transformer-based architectures) to extract the most promising approaches
- optimized, debugged and trained the architecture on a small dataset of ionic liquids
- scaled the architecture to the very large ANI-1 dataset (~21M molecule configurations)
- while optimizing for efficiency of the architecture and achieving a 4x speedup over the supervisors' baseline with comparable accuracy.
