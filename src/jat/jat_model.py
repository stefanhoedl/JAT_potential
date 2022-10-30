# Copyright 2022 Stefan Hödl

import jax
import jax.nn
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal, lecun_normal

from jraph._src.graph import GraphsTuple
from jraph._src.utils import segment_softmax

from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
from typing import Any, Callable, Sequence, Tuple
from datetime import datetime

class JatCore(nn.Module):
    """ Core of the JAT model, which incrementally refines the type embeddings
        into expressive latent feature vectors, from which the the energy
        prediction is obtained as the sum of individual atomic contributions.
        
        Every JAT layer applies one message passing step using the dynamic 
        linear attention function (GATv2).
        
        Afterwards, the regression predictions (atomic contributions) 
        are obtained from a [64:32:16:16:1] pyramidal readout head.
    
    Args:
        layer_dims: projection dimensionality of query and key, and thus output.
            Using the same dimensionality as the previous layer avoids inial 
            projection & skip projections. Try [48, 48, 48, 48] to start.
        n_head: number of attention heads for multi-headed attention. 1 head is
            recommended if periodic boundary conditions (cell_size) are used.
        
    Returns: 
        (n_atoms, 1) atomic contribution to the potential energy
    """
    layer_dims: Sequence[int]
    n_head: int = 1

    def setup(self):
        """ perform N message passing steps with dimensionalities layer_dims 
            initializes N JAT layers with n_head attention heads
            the last layer uses a dense(1) output without an activation function
        """
        
        print(f'JAT layers dimensionality {self.layer_dims} \
             with {self.n_head}-headed attention.')
        
        layers = []
        for i in range(len(self.layer_dims)-1):
            layer = JatLayer(self.layer_dims[i], self.n_head, readout=False)
            layers.append(layer)
        layer = JatLayer(self.layer_dims[-1], self.n_head, readout=True)
        layers.append(layer)

        self.jat_layers = layers

    def readout_head(self, graph: GraphsTuple):
        """ pyramidal readout head of dimensionality [64:32:16:16:1]
            uses Swish-1 nonlinearity, LayerNorm
            can be augmented with skip connections or SetTransformer
        """
        print(f"Node features shape before readout: {graph[0].shape}")
        x = graph[0]
        x = jnp.reshape(x, (x.shape[0], -1))

        for r_dim in [64, 32, 16, 16]:
            x_  = nn.Dense((r_dim), use_bias=False)(x)
            x   = nn.swish(nn.LayerNorm()(x_))
        out  = nn.swish(nn.Dense((1), use_bias=False)(x))
        return out

    @nn.compact
    def __call__(self, graphs_tuple: GraphsTuple, mask: jnp.array):
        """ 
        Args:
            graphs_tuple: edge list triplets of [senders, receivers, edges] 
                         in sparse format, each of length (_mask_dim)
                nodes:      (n_atoms, dim) node feature vector 
                senders:    list of sender indices of length 
                receivers:  list of receivers indices of length 
                edges:      pairwise interatomic distance for sender-receiver
            mask: Boolean mask of shape '_mask_dim', True if edge exists. 
        
        Returns: 
            (n_atoms, 1) atomic contribution to the potential energy
        """
        for step in range(len(self.layer_dims)):
            graphs_tuple = self.jat_layers[step](graphs_tuple, mask)
        
        contributions = self.readout_head(graphs_tuple)
        return contributions

class JatLayer(nn.Module):
    """ Initializes a single JAT layer, which performs one message passing step. 
        
    Args:
        dim: dimension of query & key projection for each JAT layer
        n_head: number of attention heads
        readout: If True, does not apply LayerNorm (for the final JAT layer)
        kernel_init: initialization, lecun_normal and orthogonal work well. 
    """
    dim: int
    n_head: int
    readout: bool
    kernel_init: Callable = orthogonal(column_axis = -2)

    def skip_proj(self, nodes: jnp.array):
        """ projects old nodes to new dimensionality 
            for element-wise addition of skip connection """
        
        if nodes.shape[-2] != self.n_head:
            nodes = jnp.stack([nodes], axis=-2)
        
        if nodes.shape[-1] != self.dim:
            nodes = nn.DenseGeneral((self.n_head, self.dim), axis=(-2,-1), 
                    use_bias=False, kernel_init = self.kernel_init)(nodes)

        return nodes
    
    def attention(self, query: jnp.array, key: jnp.array, edges: jnp.array):
        """ applies linear dynamic attention mechanism (GATv2) 
            for all sender-receiver-distance triplets in the edge list.
            edge list is padded up to static maximum of _mask_dim
            
            Args:
                query:  (_mask_dim, dim) array of sender features
                key:    (_mask_dim, dim) array of receiver features
                edges:  (_mask_dim, 1) array of pairwise interatomic distances
            
            Returns:
                attn_weights: (_mask_dim, 1) vector of attention weights
                    (unnormalized logits e_ij)
        """
        features = nn.swish(query + key)
        
        # multi-headed variant option: use edge features instead of powers
        # edges = jnp.reshape(jnp.stack([edges]*self.n_head, axis=-2), 
        #           (-1, self.n_head, 1)) 

        edges = jnp.reshape(jnp.stack([(jnp.power(edges, i+1)) \
                for i in range(self.n_head)], axis=0), (-1, self.n_head, 1))
        features = jnp.concatenate([features, edges], axis=-1)

        attn_weights = nn.DenseGeneral((self.n_head,1), axis=(-2,-1), \
                       use_bias=False, kernel_init = self.kernel_init)(features)
        return attn_weights

    def _ApplyJAT(self, graph: GraphsTuple, mask: jnp.array):
        """Applies JAT layer to perform one message passing step. 

        Args:
            graph: sparse GraphsTuple 
                nodes: node feature vectors 
                senders, receivers, edges: edge list 
            mask: boolean mask for the edge list, False if edge is padded
        
        Returns:
            graph: GraphsTuple with updated node features
        """

        nodes, edges, receivers, senders, _, _, _ = graph
        sum_n_node = nodes.shape[0]
        old_nodes = nodes
        
        # add head dimension in first layer for subsequent projection
        if nodes.shape[-2] != self.n_head:
            nodes = jnp.stack([nodes], axis=-2)

        # query & key projection: nodes features to layer_dims. 
        querys = nn.DenseGeneral((self.n_head, self.dim), axis=(-2,-1), \
                    use_bias=False, kernel_init = self.kernel_init)(nodes)
        keys   = nn.DenseGeneral((self.n_head, self.dim), axis=(-2,-1), \
                    use_bias=False, kernel_init = self.kernel_init)(nodes)
        
        # Lift projected sender & receiver features using edge list 
        sent_attributes = querys[senders]     
        received_attributes = keys[receivers] 
        
        # calculate attention weights
        attn_weights = self.attention(sent_attributes, 
                                received_attributes, 
                                edges)

        # apply mask
        inf_mask = jnp.reshape(jnp.where(mask, 0, -jnp.inf), (-1, 1, 1))      
        attn_weights = attn_weights + inf_mask
        
        # normalize weights using segment softmax, vmap across heads
        weights = jax.vmap(
            segment_softmax,
            in_axes=(1, None, None), out_axes=(1)
        )(attn_weights, receivers, sum_n_node)

        # Apply weights
        messages = sent_attributes * weights
        
        # Aggregate weighted messages to receiving nodes, vmap across heads
        new_nodes = jax.vmap(
            jax.ops.segment_sum,
            in_axes=(1, None, None), out_axes=(1)
        )(messages, receivers, sum_n_node)
        
        new_nodes = nn.swish(new_nodes)
        new_nodes = new_nodes + self.skip_proj(old_nodes)

        ## if not final layer (readout), apply LayerNorm
        if not self.readout:
            new_nodes = nn.LayerNorm()(new_nodes)

        return graph._replace(nodes=new_nodes)
    
    @nn.compact
    def __call__(self, graphs_tuple, mask):
        return self._ApplyJAT(graphs_tuple, mask)


class GraphGenerator:
    """ Generates a GraphsTuple describing the configuration as a graph,
        where atoms seperated by less than 'cutoff' share an edge.
        Uses a Sparse Matrix representation ('edge list') with masked edges
        
        Args:
            n_atoms: size of the system
            cutoff: The maximum distance for two atoms to be considered 
                    neighbors (graph cutoff), 
                    or None for fully connected graph (n_atoms**2 edges)
            cell_size: Unit cell vector matrix (3x3) if the system is periodic.
            max_neighbors: to calculate upper bound for mask_dim
    """

    def __init__(self, 
        n_atoms: int, 
        cutoff: float, 
        cell_size: float, 
        max_neighbors: int,
    ):
        self.n_atoms = n_atoms
        self.cutoff = cutoff
        self.cell_size = cell_size
        self.max_neighbors = max_neighbors
        
        if max_neighbors is not None:
            self._mask_dim = self.n_atoms * self.max_neighbors
        else:
            self._mask_dim = self.n_atoms ** 2
        
    def make_graph(self, coordinates):
        """        
        Args:
            coordinates: An (n_atoms, 3) array of atomic positions.

        Returns:
            GraphsTuple: edge list triplets of [senders, receivers, edges] 
                         in sparse format, each of length (_mask_dim)
                nodes:      (n_atoms, dim) node feature vector 
                senders:    list of sender indices of length 
                receivers:  list of receivers indices of length 
                edges:      pairwise interatomic distance for sender-receiver
            mask: Boolean mask of shape '_mask_dim', True if edge exists.
        """
        cutoff2 = self.cutoff * self.cutoff
        delta = coordinates - coordinates[:, jnp.newaxis, :]
        
        # calculate distances for cell box with periodic boundary conditions
        if self.cell_size is not None:
            delta -= jnp.einsum(
                "ijk,kl",
                jnp.round(jnp.einsum("ijk,kl", delta, 
                    jnp.linalg.pinv(self.cell_size))
                ),
                self.cell_size
            )
        distances2 = jnp.sum(delta**2, axis=2)

        # filter edges where pairwise distance < graph cutoff
        indices = jnp.where(distances2 < cutoff2, size=self._mask_dim, \
                    fill_value=self._mask_dim + 1)        
        edges = distances2[indices]
        
        # mask edges up to a static maximum 
        mask = jnp.where(indices[0] == self._mask_dim + 1, False, True)

        nodes = jnp.ones(coordinates.shape[0]) #placeholder
        senders = indices[0]
        receivers = indices[1]
        
        return GraphsTuple(
            nodes=nodes.astype(jnp.float32),
            n_node=jnp.reshape(nodes.shape[0], [1]),
            edges=edges.astype(jnp.float32),
            n_edge=jnp.reshape(edges.shape[0], [1]),
            globals=jnp.zeros((1, 1), dtype=jnp.float32),
            receivers=receivers.astype(jnp.int32),
            senders=senders.astype(jnp.int32)
            ), mask

    def __call__(
        self, 
        coordinates: jnp.ndarray,
    ) -> Tuple[GraphsTuple, jnp.ndarray]:
        return self.make_graph(coordinates)


class JatModel(nn.Module):
    """Wrapper model around the core layers to calculate energies and forces.

    The class does not provide a __call__ method, forcing the user to choose
    what to evaluate (forces, energies or both).

    Args:
        n_types: The number of atomic types in the system.
        embed_d: The dimension of the embedding vector. 
        graph_generator: The function mapping atomic coordinates to a 
            GraphsTuple which represents the system using a sparse edge list.
        core_model: The model that takes the generated graph and embedded 
            features and returns the atomic contributions to the energy.
    """
    n_types: int
    embed_d: int
    graph_generator: Callable
    core_model: nn.Module

    def setup(self):
        # These neurons create the embedding vector.
        self.embed = nn.Embed(self.n_types, self.embed_d)
        # This linear layer centers and scales the energy after the core
        # has done its job.
        self.denormalizer = nn.Dense(1)
        # The checkpointing strategy can be reconsidered to achieve different
        # tradeoffs between memory and CPU usage.
        self._calc_gradient = jax.checkpoint(
            jax.grad(self.calc_potential_energy, argnums=0)
        )
        self._calc_value_and_gradient = jax.checkpoint(
            jax.value_and_grad(self.calc_potential_energy, argnums=0)
        )


    def calc_atomic_energies(self, positions, types, cell):
        """Compute the atomic contributions to the potential energy.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.
            
        Returns:
            The n_atoms contributions to the energy.
        """
        graph, mask = self.graph_generator(positions)
        embeddings = self.embed(types)
        graph = graph._replace(nodes = embeddings)

        contributions = self.core_model(graph, mask)
        contributions = self.denormalizer(contributions)
        return contributions

    def calc_potential_energy(self, positions, types, cell):
        """Compute the total potential energy of the system.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The sum of all atomic contributions to the potential energy.
        """
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        """Compute the force on each atom.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The (n_atoms, 3) vector containing all the forces.
        """
        return -self._calc_gradient(positions, types, cell)

    def calc_potential_energy_and_forces(self, positions, types, cell):
        """Compute the total potential energy and all the forces.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atom types, codified as integers from 0 to n_types - 1.
            cell: The (3, 3) matrix representing the simulation box if periodic
                boundary conditions are in effect. If it is not periodic along
                one or more directions, signal that fact with one of more zero
                vectors.

        Returns:
            A two-element tuple. The first element is the sum of all atomic
            contributions to the potential energy. The second one is an
            (n_atoms, 3) vector containing all the forces.
        """
        energy, gradient = self._calc_value_and_gradient(positions, types, cell)
        return (energy, -gradient)


@dataclass
class JATModelInfo:
    # A description of the general class of model
    model_name: str
    # Details about the dataset/subset trained on
    model_details: str
    # A datetime object with the time of training
    timestamp: datetime
    # A cutoff radius for the graph generator
    graph_cut: float
    # Maximum number of neighbors 
    graph_neighbors: int
    # Alphabetical list of element symbols
    sorted_elements: list
    # Dimensionality of the element embedding
    embed_d: int
    # List of widths of the core layers
    layer_dims: list
    # Number of attention n_head
    n_head: int
    # Number of atoms to consider
    n_atoms: int
    # Dictionary of additional arguments to the model constructor
    constructor_kwargs: dict
    # Random seed used to create the RNG for training
    random_seed: int
    # Dictionary of model parameters created by flax
    params: FrozenDict
    # Any other information this kind of model requires
    specific_info: Any

if __name__ == "__main__":
    import collections
    import jax
    import jax.numpy as jnp
    import pathlib
    import json
    from jax.config import config 
    config.update("jax_debug_nans", True) 

    N_PAIR = 15
    CONFIGS_DFT = "../../ean/configurations.json" 
    
    cells = []
    positions = []
    energies = []
    forces = []
    
    with open(
    (pathlib.Path(__file__).parent /
    CONFIGS_DFT).resolve(), "r") as json_dft: 
        for line in json_dft:
            while len(cells) < 8:
                json_data = json.loads(line)
                cells.append(jnp.diag(jnp.array(json_data["Cell-Size"]))) 
                positions.append(json_data["Positions"])
                energies.append(json_data["Energy"])
                forces.append(json_data["Forces"])
    
    GRAPH_CUT = 5
    EMBED_D = 48
    LAYER_DIMS = [48, 48, 48, 48]
    N_HEADS = 1

    type_cation = ["N", "H", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
    type_anion = ["N", "O", "O", "O"]
    types = N_PAIR * type_cation + N_PAIR * type_anion
    
    unique_types = sorted(set(types))
    type_dict = collections.OrderedDict()
    for i, k in enumerate(unique_types):
        type_dict[k] = i
    types = jnp.array([type_dict[i] for i in types])
    n_atoms = len(types)

    n_configurations = len(positions)
    types = jnp.array([types for i in range(n_configurations)])

    cells = jnp.array(cells)
    positions = jnp.array(positions)
    energies = jnp.array(energies)
    forces = jnp.array(forces)

    graph_neighbors = 63
    n_types = int(types.max()) + 1

    core_model = JatCore(layer_dims = LAYER_DIMS, n_head = N_HEADS)
    graph_gen = GraphGenerator(n_atoms, GRAPH_CUT, cells[0], graph_neighbors)
    dynamics_model = JatModel(n_types, EMBED_D, graph_gen, core_model)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    params = dynamics_model.init(
        init_rng,
        positions[0],
        types[0],
        cells[0],
        method=JatModel.calc_forces
    )

    pred = dynamics_model.apply(
                params,
                positions[0],
                types[0],
                cells[0],
                method=JatModel.calc_forces
            )

    rmse = jnp.sqrt( jnp.mean( (pred - forces[0])**2 ) )
    mae = jnp.mean( (pred - forces[0]) )
    print(f"MAE {mae} eV / Å,  RMSE {rmse} eV / Å, ")