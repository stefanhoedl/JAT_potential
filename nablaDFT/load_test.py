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

from jat.jat_model import JatCore, JatModel, GraphGenerator, JATModelInfo
from jat.training import *
from jat.utilities import create_array_shuffler, draw_urandom_int32, \
    get_max_number_of_neighbors

from nablaDFT.nablaDFT.dataset import HamiltonianDatabase

train = HamiltonianDatabase("dataset_train_2k.db")
Z, R, E, F, H, S, C = train[0]  # atoms numbers, atoms positions, energy, forces, core hamiltonian, overlap matrix, coefficients matrix


from ase.db import connect

train = connect("train_2k_energy.db")
atoms_data = connect.get(1)

SEED = 42

rng = jax.random.PRNGKey(SEED)
instance_code = draw_urandom_int32()
PICKLE_FILE = f'./ani1/models/JAT_ANI_{instance_code}.pickle'

# loop through dataset once to collect unique elements and maximum atoms
max_nbr_atoms = 0
sorted_elements = set()
for file_num in SUBSET:
    hdf5file = f'./ani1/ANI-1_release/ani_gdb_s0' + str(file_num) + '.h5'
    # adl = pya.anidataloader(hdf5file)
    total_nmolecules = 0
    for data in adl:
        t = data['species']
        e = data['energies']
        total_nmolecules += 1
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
positions_test = np.zeros_like(positions)
energies_test = np.zeros_like(energies)
types_test = np.zeros_like(types)