import sys
import os
import pathlib
import json
import pickle
from datetime import datetime
from collections import OrderedDict
import jax
import jax.numpy as jnp
import flax
import numpy as np

if os.getcwd().startswith('/workspace/'):
    ## docker on rig
    sys.path.append('/workspace/JAT_potential/src')
    log_wandb = True
else:
    ## local
    sys.path.append('/home/stefan/jobs/JAT_potential/src')
    log_wandb = False

from jat.jat_model import JatCore, JatModel, GraphGenerator, JATModelInfo
from jat.training import *
from jat.utilities import create_array_shuffler, draw_urandom_int32, \
    get_max_number_of_neighbors

from nablaDFT.nablaDFT.dataset import HamiltonianDatabase

train = HamiltonianDatabase("./nablaDFT/data/dataset_train_2k.db")
test = HamiltonianDatabase("./nablaDFT/data/dataset_test_2k_conformers.db")
Z, R, E, F, _, _, _ = train[0]  # atoms numbers, atoms positions, energy, forces, core hamiltonian, overlap matrix, coefficients matrix
print(len(train))
print(f"Z {Z.shape}, R {R.shape}, E {E.shape}, F {F.shape}")

SEED = 42

rng = jax.random.PRNGKey(SEED)
instance_code = draw_urandom_int32()
PICKLE_FILE = f'./nablaDFT/models/JAT_nabla_{instance_code}.pickle'

# loop through dataset once to collect unique elements and maximum atoms
max_nbr_atoms = 0
sorted_elements = set()
for i in range(len(train)):
    t, _, _, _, _, _, _ = train[i] # types
    max_nbr_atoms = np.max((max_nbr_atoms, len(t)))
    sorted_elements = sorted_elements.union(set(t))

# create element --> int mapping dict
type_dict = OrderedDict()
sorted_elements = sorted(sorted_elements)
for i, k in enumerate(sorted_elements):
    type_dict[k] = i

# create numpy arrays for positions, types & energies (p, t, e)
positions = np.empty((0, max_nbr_atoms, 3))
types = np.empty((0, max_nbr_atoms), dtype=int)
energies = np.empty(0)
forces = np.empty((0, max_nbr_atoms, 3))

# loop through molecules
subtotal = 0
for i in range(len(train)):
    rng, data_rng = jax.random.split(rng)
    _t, p, e, f, _, _, _ = train[i]
    t = np.array([type_dict[i] for i in _t], dtype=int)
    # t: types
    # p: Cartesian coordinates in bohr
    # e: energy in Eh
    # f: forces in Eh/bohr

    # pad up to static maximum (max_nbr_atoms)
    padding = max_nbr_atoms - p.shape[0]
    if padding:
        p = np.pad(p, ((0, padding), (0, 0)), "constant")
        t = np.pad(t, (0, padding), "constant", constant_values=-1)
        f = np.pad(f, ((0, padding), (0, 0)), "constant")

    positions = np.append(positions, p[None, ...], axis=0)
    energies = np.append(energies, e[...], axis=0)
    forces = np.append(forces, f[None, ...], axis=0)
    types = np.append(types, t[None, ...], axis=0)

    # break if MAX_CONFIGS exceeded
    subtotal += 1
    if subtotal >= 5000:
        break
print(positions.shape, energies.shape, forces.shape, types.shape)