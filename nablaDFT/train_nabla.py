#!/usr/bin/env python
# Copyright 2022 Stefan Hödl
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

# Training Config
TRAINING_FRACTION = .9
N_BATCH = 8
N_EVAL_BATCH = 32
SEED = 42
LOG_COSH_PARAMETER = 1e0  # In angstrom / eV
LR_MIN, LR_MAX, LR_END = 0.5e-4, 0.3e-3, 0.5e-6
N_EPOCHS = 21

# JAT MODEL
LAYER_DIMS = [48, 48, 48, 48]
GRAPH_CUT = 5
EMBED_D = 48
N_HEADS = 1

rng = jax.random.PRNGKey(SEED)
instance_code = draw_urandom_int32()
PICKLE_FILE = f"./nablaDFT/models/JAT_nablaDFT_{instance_code}.pickle"

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

# loop through molecules: train
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
    if subtotal >= 20000:
        break
print(positions.shape, energies.shape, forces.shape, types.shape)

# loop through molecules: test
positions_test = np.empty((0, max_nbr_atoms, 3))
types_test = np.empty((0, max_nbr_atoms), dtype=int)
energies_test = np.empty(0)
forces_test = np.empty((0, max_nbr_atoms, 3))
for i in range(len(test)):
    rng, data_rng = jax.random.split(rng)
    _t, p, e, f, _, _, _ = test[i]
    t = np.array([type_dict[i] for i in _t], dtype=int)
    
    # pad up to static maximum (max_nbr_atoms)
    padding = max_nbr_atoms - p.shape[0]
    if padding:
        p = np.pad(p, ((0, padding), (0, 0)), "constant")
        t = np.pad(t, (0, padding), "constant", constant_values=-1)
        f = np.pad(f, ((0, padding), (0, 0)), "constant")

    positions_test = np.append(positions_test, p[None, ...], axis=0)
    energies_test = np.append(energies_test, e[...], axis=0)
    forces_test = np.append(forces_test, f[None, ...], axis=0)
    types_test = np.append(types_test, t[None, ...], axis=0)

n_configurations = len(positions)
n_train = int(TRAINING_FRACTION * n_configurations)
n_validate = (n_configurations - n_train)

n_types = len(sorted_elements)
cells = np.stack([np.eye(3)*0.] * n_configurations, axis=0)
cells_test = np.stack([np.eye(3)*0.] * len(positions_test), axis=0)

def split_array(in_array):
    "Split an array in training and validation sections."
    return jnp.split(in_array, (n_train, n_train + n_validate))[:2]

rng, shuffler_rng = jax.random.split(rng)
shuffle = create_array_shuffler(shuffler_rng)

positions_train, positions_validate = split_array(shuffle(positions))
types_train, types_validate = split_array(shuffle(types))
energies_train, energies_validate = split_array(shuffle(energies))
forces_train, forces_validate = split_array(shuffle(forces))
cells_train, cells_validate = split_array(shuffle(cells))

# determine maximum number of neighbors
graph_neighbors = 1
for p, t, c in zip(positions, types, cells):
    graph_neighbors = max(
        graph_neighbors,
        get_max_number_of_neighbors(
            jnp.asarray(p),
            jnp.asarray(t),
            GRAPH_CUT,
            jnp.asarray(c)))
print(f"Maximum of {graph_neighbors} neighbors for Graph generation")

# Call JAT model & components constructors
core_model = JatCore(
    LAYER_DIMS,
    N_HEADS
)
graph_gen = GraphGenerator(
    max_nbr_atoms,
    GRAPH_CUT,
    None,
    graph_neighbors
)
dynamics_model = JatModel(
    n_types,
    EMBED_D,
    graph_gen,
    core_model
)

# Initialize the JAT model parameters
rng, init_rng = jax.random.split(rng)
model_params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    None,
    method=JatModel.calc_forces
)

# Create and initialize the one cycle minimizer
optimizer = create_one_cycle_minimizer(
    n_train // N_BATCH, LR_MIN, LR_MAX, LR_END
)
optimizer_state = optimizer.init(model_params)

# Create the function that will compute the contribution to the loss from a
# single data point. In this case, our loss will not make use of the energies.
log_cosh = create_log_cosh(LOG_COSH_PARAMETER)

if log_wandb:
    import wandb
    wandb.init(project='jat-nablaDFT', config={
        "N_EPOCHS": N_EPOCHS,
        "TRAINING_FRACTION": TRAINING_FRACTION,
        "GRAPH_CUT": GRAPH_CUT,
        "PICKLE_FILE": PICKLE_FILE,
        "N_BATCH": N_BATCH,
        "LOG_COSH_PARAMETER": LOG_COSH_PARAMETER,
        "SEED": SEED,
        "layer_dims": LAYER_DIMS,
        "N_HEADS": N_HEADS,
        "LR_MIN": LR_MIN,
        "LR_MAX": LR_MAX,
        "LR_END": LR_END,
        "instance": instance_code,
    })
    config = wandb.config

# Get flattened key-value list of trainable parameters.
flat_params = {'/'.join(k[-2:]): v.shape for k, v in
                    flax.traverse_util.flatten_dict(
                        flax.core.unfreeze(model_params)).items()}
print(flat_params)

def calc_loss_contribution(pred_energy, pred_forces, obs_energy, obs_forces):
    "Return the log-cosh of the difference between predicted and actual forces"
    delta_forces = obs_forces - pred_forces
    return log_cosh(delta_forces).mean()


# Create a driver for each training step.
training_step = create_training_step(
    dynamics_model, optimizer, calc_loss_contribution
)

rng, epoch_rng = jax.random.split(rng)
training_epoch = create_training_epoch(
    positions_train,
    types_train,
    cells_train,
    energies_train,
    forces_train,
    N_BATCH,
    training_step,
    epoch_rng,
    log_wandb=True
)

# Create a dictionary of validation statistics that we want calculated.
validation_statistics = {
    "force_MAE": fmae_validation_statistic,
    "energy_MAE": emae_validation_statistic,
    "force_RMSE": frmse_validation_statistic,
    "energy_RMSE": ermse_validation_statistic
}
validation_units = {
    "force_MAE": "eV / Å",
    "energy_MAE": "eV / atom",
    "force_RMSE": "eV / Å",
    "energy_RMSE": "eV / atom"
}

# Create the driver for the validation step.
validation_step = create_validation_step(
    dynamics_model,
    validation_statistics,
    positions_validate,
    types_validate,
    cells_validate,
    energies_validate,
    forces_validate,
    N_EVAL_BATCH,
    progress_bar=False
)
test_step = create_validation_step(
    dynamics_model,
    validation_statistics,
    positions_test,
    types_test,
    cells_test,
    energies_test,
    forces_test,
    N_EVAL_BATCH,
    progress_bar=False
)

min_mae = jnp.inf
for i in range(N_EPOCHS):
    # Reset the training schedule and run a full epoch.
    optimizer_state = reset_one_cycle_minimizer(optimizer_state)
    optimizer_state, model_params = training_epoch(
        optimizer_state, model_params)

    # Evaluate the results.
    statistics = validation_step(model_params)
    mae = statistics["force_MAE"]
    rmse = statistics["force_RMSE"]
    test_statistics = test_step(model_params)
    test_mae = test_statistics["force_MAE"]
    test_rmse = test_statistics["force_RMSE"]

    # Print the relevant statistics.
    print(
        f"VAL  RMSE = {rmse} {validation_units['force_RMSE']}. "
        f"VAL  MAE  = {mae} {validation_units['force_MAE']}."
        f"TEST RMSE = {test_rmse} {validation_units['force_RMSE']}. "
        f"TEST MAE  = {test_mae} {validation_units['force_MAE']}."
    )
    if log_wandb:
        wandb.log({"rmse": rmse.copy(), "mae": mae.copy()}, commit=False)
        wandb.log({"test_rmse": test_rmse.copy(), "test_mae": test_mae.copy()})

    # Save the state only if the validation MAE is minimal.
    if mae < min_mae:
        min_mae = mae
        model_info = JATModelInfo(
            model_name="JAT",
            model_details=f"nablaDFT 2k train test",
            timestamp=datetime.now(),
            n_atoms=max_nbr_atoms,
            graph_cut=GRAPH_CUT,
            graph_neighbors=graph_neighbors,
            sorted_elements=sorted_elements,
            embed_d=EMBED_D,
            layer_dims=LAYER_DIMS,
            n_head=N_HEADS,
            constructor_kwargs={"cell_size": None},
            random_seed=SEED,
            params=flax.serialization.to_state_dict(model_params),
            specific_info=None)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(model_info, f, protocol=5)
        print(f"woooo {mae:.4f} mae & {rmse:.4f} rmse")

    # Periodically save the best model (every 10 epochs).
    if (i % 10) == 9 and i > 0:
        PICKLE_EPOCH = \
            f'./nablaDFT/models/JAT_nablaDFT{instance_code}_epoch{i+1}.pickle'
        with open(PICKLE_EPOCH, "wb") as f:
            pickle.dump(model_info, f, protocol=5)


###################################################
# # Load trained model & evaluate test set once again
with open(PICKLE_FILE, "rb") as f:
    model_info = pickle.load(f)

core_model = JatCore(
    layer_dims=model_info.layer_dims,
    n_head=model_info.n_head
)
graph_gen = GraphGenerator(
    model_info.n_atoms,
    model_info.graph_cut,
    None,
    model_info.graph_neighbors
)
dynamics_model = JatModel(
    len(model_info.sorted_elements),
    model_info.embed_d,
    graph_gen,
    core_model
)

# Initialize model weights
template_params = dynamics_model.init(
    init_rng,
    jnp.zeros((model_info.n_atoms, 3), dtype=jnp.float32),
    jnp.zeros(model_info.n_atoms, dtype=jnp.int32),
    jnp.eye(3),
    method=JatModel.calc_forces
)
# Load best model parameters to model
model_params = flax.serialization.from_state_dict(
    template_params,
    model_info.params)

statistics = validation_step(model_params)
mae = statistics["force_MAE"]
rmse = statistics["force_RMSE"]
test_statistics = test_step(model_params)
test_mae = test_statistics["force_MAE"]
test_rmse = test_statistics["force_RMSE"]

print(
    f"Validation & Test set statistics with loaded parameters: \n"
    f"VAL RMSE = {rmse:.4f} {validation_units['force_RMSE']}. \n"
    f"VAL MAE = {mae:.4f} {validation_units['force_MAE']}. \n"
    f"TEST RMSE = {test_rmse:.4f} {validation_units['force_RMSE']}. \n"
    f"TEST = {test_mae:.4f} {validation_units['force_MAE']}."
)
