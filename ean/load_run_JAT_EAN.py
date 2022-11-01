#!/usr/bin/env python
# Copyright 2022 Stefan Hödl
import collections
import jax
import jax.numpy as jnp
import flax
import pathlib
import json
import sys
import pickle
from collections import OrderedDict

sys.path.append('/workspace/JAT_potential/src')
from jat.jat_model import JatCore, JatModel, GraphGenerator, JATModelInfo

N_PAIR = 15
CONFIGS_DFT = "../ean/configurations.json" 
PICKLE_FILE = f"./ean/models/JAT_EAN15_ep3K.pickle"

cells = []
positions = []
energies = []
forces = []

with open(
(pathlib.Path(__file__).parent /
CONFIGS_DFT).resolve(), "r") as json_dft: 
    for line in json_dft:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell-Size"]))) 
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])

GRAPH_CUT = 5
EMBED_D = 48
LAYER_DIMS = [48, 48, 48, 48]
N_HEADS = 1

with open(PICKLE_FILE, "rb") as f:
    model_info = pickle.load(f)

type_cation = ["N", "H", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
type_anion = ["N", "O", "O", "O"]
types = N_PAIR * type_cation + N_PAIR * type_anion

type_dict = OrderedDict()
for i, k in enumerate(model_info.sorted_elements):
    type_dict[k] = i
types = jnp.array([type_dict[i] for i in types])

n_configurations = len(positions)
types = jnp.array([types for i in range(n_configurations)])

cells = jnp.array(cells)
positions = jnp.array(positions)
energies = jnp.array(energies)
forces = jnp.array(forces)

graph_neighbors = 63
n_types = int(types.max()) + 1

core_model = JatCore(layer_dims = model_info.layer_dims, 
                    n_head=model_info.n_head)
graph_gen = GraphGenerator(model_info.n_atoms, 
                    model_info.graph_cut, 
                    model_info.constructor_kwargs["cell_size"], 
                    model_info.graph_neighbors)
dynamics_model = JatModel(
                    len(model_info.sorted_elements), 
                    model_info.embed_d, 
                    graph_gen, 
                    core_model)

rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng)

# Initialize model weights
template_params = dynamics_model.init(
    init_rng,
    jnp.zeros((model_info.n_atoms, 3), dtype=jnp.float32),
    jnp.zeros(model_info.n_atoms, dtype=jnp.int32),
    jnp.eye(3),
    method=JatModel.calc_forces
)
# Load best model parameters to model
model_params = flax.serialization.from_state_dict(template_params, 
    model_info.params)

pred = dynamics_model.apply(
            model_params,
            positions[0],
            types[0],
            cells[0],
            method=JatModel.calc_forces
        )

rmse = jnp.sqrt( jnp.mean( (pred - forces[0])**2 ) )
mae = jnp.mean( (pred - forces[0]) )
print(f"MAE {mae:.4f} eV / Å,  RMSE {rmse:.4f} eV / Å, ")