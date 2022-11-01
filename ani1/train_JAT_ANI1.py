#!/usr/bin/env python
import sys
import pickle
from collections import OrderedDict
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.optim
from datetime import datetime
import time

import pyanitools as pya
from jat.jat_model import JatCore, JatModel, GraphGenerator, JATModelInfo
from jat.training import *
from jat.utilities import create_array_shuffler, draw_urandom_int32, \
    get_max_number_of_neighbors

log_wandb = True

# Training config
N_BATCH = 32
N_EVAL_BATCH = 128
SEED = 42
LOG_COSH_PARAMETER = 1e-1  # In angstrom / eV
LR_MIN, LR_MAX, LR_END = 1e-4, 1e-3, 1e-5
N_EPOCHS = 30

# Data config
# Load up to MAX_CONFIGS conformations per subsets with N heavy atoms
# Sample CONF_FRAC of conformations and allocate to test set with
# probability TEST_FRACTION, otherwise to train+validation set.
# Split into Train & Validation with TRAINING_FRACTION.

# subset 1 to 7 with MAX <50K:  ~220k train, 25k test, 614/59 molecules
# subset 1 to 7 with MAX <100K: ~380k train, 46k test, 1178/143 molecules
# subset 1 to 7 with MAX <200K: ~588k train, 69k test, 2369/260 molecules

SUBSET = [1, 2, 3, 4, 5, 6, 7]
MAX_CONFIGS = 50000
CONF_FRAC = 0.05
TRAINING_FRACTION = 0.9
TEST_FRACTION = 0.1

# JAT model config
GRAPH_CUT = 3
EMBED_D = 48
LAYER_DIMS = [48, 48, 48, 48, 48]
N_HEADS = 4

rng = jax.random.PRNGKey(SEED)
instance_code = draw_urandom_int32()
PICKLE_FILE = f'./ani1/models/JAT_ANI_{instance_code}.pickle'

# loop through dataset once to collect unique elements and maximum atoms
max_nbr_atoms = 0
sorted_elements = set()
for file_num in SUBSET:
    hdf5file = f'./ani1/ANI-1_release/ani_gdb_s0' + str(file_num) + '.h5'
    adl = pya.anidataloader(hdf5file)
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

# loop through subsets and store p, t, e in train+val or test set arrays
for file_num in SUBSET:
    hdf5file = f'./ani1/ANI-1_release/ani_gdb_s0' + str(file_num) + '.h5'
    adl = pya.anidataloader(hdf5file)
    subtotal = 0
    nmolecules = 0
    nmolecules_test = 0
    test_molecules = []

    # loop through molecules
    for data in adl:
        rng, data_rng = jax.random.split(rng)
        p = data['coordinates']
        _t = data['species']
        e = data['energies']
        t = np.array([type_dict[i] for i in _t], dtype=int)
        nconfigs = len(e)

        # pad up to static maximum (max_nbr_atoms)
        padding = max_nbr_atoms - p.shape[1]
        if padding:
            p = np.pad(p, ((0, 0), (0, padding), (0, 0)), "constant")
            t = np.pad(t, (0, padding), "constant", constant_values=-1)

        # skip all c2h6n2 isomers for separate evaluation
        _tm = np.unique(_t, return_counts=True)
        if ('N' in _tm[0]) and ('C' in _tm[0]) \
                and ('H' in _tm[0]) and not ('O' in _tm[0]):
            if (_tm[1] == [2, 6, 2]).all():
                continue
            elif (_tm[1] == [4, 4, 4]).all():
                continue

        # Draw 5% of conformation indices (or CONF_FRAC %).
        # Allocate to test set with prob. TEST_FRACTION, else to train&val set.
        idx = jax.random.choice(
            data_rng,
            nconfigs,
            shape=[round(nconfigs*CONF_FRAC)+1],
            replace=False)
        if jax.random.choice(data_rng, 100) >= TEST_FRACTION*100:
            subtotal += len(idx)
            nmolecules += 1
            positions = np.append(positions, p[idx, ...], axis=0)
            energies = np.append(energies, e[idx, ...], axis=0)
            types = np.append(types, np.stack([t]*len(idx)), axis=0)
        else:
            nmolecules_test += 1
            test_molecules.append(_t)
            positions_test = np.append(positions_test, p[idx, ...], axis=0)
            energies_test = np.append(energies_test, e[idx, ...], axis=0)
            types_test = np.append(types_test, np.stack([t]*len(idx)), axis=0)

        # break if MAX_CONFIGS exceeded
        if subtotal >= MAX_CONFIGS:
            break
    adl.cleanup()

print(f"pos: {positions.shape}, test {positions_test.shape}")
print(f"types: {types.shape}, test {types_test.shape}")
print(f"energies: {energies.shape}, test {energies_test.shape}")
print(f"molecules: {nmolecules} in train&val, {nmolecules_test} test only")

n_configurations = len(positions)
n_train = int(TRAINING_FRACTION * n_configurations)
n_validate = (n_configurations - n_train)

# create empty cells (PBC box) since no PBC are in use
n_types = len(sorted_elements)
cells = np.stack([np.eye(3)*0.] * n_configurations, axis=0)
cells_test = np.stack([np.eye(3)*0.] * len(positions_test), axis=0)
forces_test = np.zeros_like(positions_test)

# shuffle and split array into train, validation
rng, shuffler_rng = jax.random.split(rng)
shuffle = create_array_shuffler(shuffler_rng)

def split_array(in_array):
    "Split an array in training and validation sections."
    return jnp.split(in_array, (n_train, n_train + n_validate))[:2]


cells_train, cells_validate = split_array(shuffle(cells))
positions_train, positions_validate = split_array(shuffle(positions))
types_train, types_validate = split_array(shuffle(types))
energies_train, energies_validate = split_array(shuffle(energies))
forces_train, forces_validate = split_array(np.zeros_like(positions))
print("*** done loading data***")

# loop through positions to evaluate maximum number of neighbours
graph_neighbors = 1
for p, t in zip(positions, types):
    graph_neighbors = max(
        graph_neighbors,
        get_max_number_of_neighbors(
            np.asarray(p),
            np.asarray(t),
            GRAPH_CUT,
            None
        )
    )
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
    method=JatModel.calc_potential_energy
)

# Create and initialize the one cycle minimizer.
optimizer = create_one_cycle_minimizer(
    n_train // N_BATCH, LR_MIN, LR_MAX, LR_END
)
optimizer_state = optimizer.init(model_params)

# Create the loss function (log-cosh)
log_cosh = create_log_cosh(LOG_COSH_PARAMETER)

if log_wandb:
    import wandb
    wandb.init(project='JAT-ANI', config={
        "SUBSET": SUBSET,
        "N_EPOCHS": N_EPOCHS,
        "TRAINING_FRACTION": TRAINING_FRACTION,
        "TEST_FRACTION": TEST_FRACTION,
        "GRAPH_CUT": GRAPH_CUT,
        "PICKLE_FILE": PICKLE_FILE,
        "N_BATCH": N_BATCH,
        "LOG_COSH_PARAMETER": LOG_COSH_PARAMETER,
        "SEED": SEED,
        "LAYER_DIMS": LAYER_DIMS,
        "N_HEADS": N_HEADS,
        "LR_MIN": LR_MIN,
        "LR_MAX": LR_MAX,
        "LR_END": LR_END,
        "INSTANCE": instance_code,
        "MAX_CONFIGS": MAX_CONFIGS,
        "CONF_FRAC": CONF_FRAC
    })
    config = wandb.config

# Print flatten key-value list of trainable parameter shapes
flat_params = {'/'.join(k[-2:]): v.shape for k, v in
                    flax.traverse_util.flatten_dict(
                        flax.core.unfreeze(model_params)).items()}
print(flat_params)
print(f'{graph_neighbors} max. graph neighbors')

# Print train shapes & n_molecules for logging
print(f"total number of molecules: {nmolecules} in train")
print(f"train pos {positions_train.shape}, val {positions_validate.shape}")
print(f"test  pos {positions_test.shape} with {nmolecules_test}  molecules")

def calc_loss_contribution(pred_energy, pred_forces, obs_energy, obs_forces):
    "Return the log-cosh of the difference between predicted and actual energy"
    delta_energy = obs_energy - pred_energy
    return log_cosh(delta_energy).mean()


# Create a driver for training steps and epoch.
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
    log_wandb=log_wandb
)

# Create a dictionary of validation statistics.
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

# Create the driver for the validation and test step.
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
start = time.time()
for i in range(N_EPOCHS):
    # Reset the training schedule and run a full training epoch.
    optimizer_state = reset_one_cycle_minimizer(optimizer_state)
    optimizer_state, model_params = training_epoch(
        optimizer_state, model_params
    )

    # Evaluate the results, print & log the relevant statistics.
    statistics = validation_step(model_params)
    mae = statistics["energy_MAE"]
    rmse = statistics["energy_RMSE"]
    test_statistics = test_step(model_params)
    test_mae = test_statistics["energy_MAE"]
    test_rmse = test_statistics["energy_RMSE"]
    print(
        f"VAL  MAE  = {mae:.4f} {validation_units['force_MAE']},"
        f"RMSE = {rmse:.4f} {validation_units['force_RMSE']}. \n"
        f"TEST MAE  = {test_mae:.4f} {validation_units['force_MAE']}, "
        f"RMSE = {test_rmse:.4f} {validation_units['force_RMSE']}."
    )

    if log_wandb:
        elapsed = time.time() - start
        wandb.log({
            "rmse": rmse.copy(),
            "mae": mae.copy(),
            "test rmse": test_rmse.copy(),
            "test mae": test_mae.copy(),
            "time/cumulative_time_mins": elapsed/60})

    # Save the state only if the validation MAE is minimal.
    if mae < min_mae:
        min_mae = mae
        model_info = JATModelInfo(
            model_name="IL_JAT_ANI",
            model_details=f"ANI {max(SUBSET)}",
            timestamp=datetime.now(),
            graph_cut=GRAPH_CUT,
            graph_neighbors=graph_neighbors,
            sorted_elements=sorted_elements,
            embed_d=EMBED_D,
            layer_dims=LAYER_DIMS,
            n_head=N_HEADS,
            n_atoms=max_nbr_atoms,
            constructor_kwargs={"cell_size": None},
            random_seed=SEED,
            params=flax.serialization.to_state_dict(model_params),
            specific_info=None
        )
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(model_info, f, protocol=5)
        print(f"woooo {mae:.4f} mae & {rmse:.4f} rmse")

    # Periodically save the best model (every 10 epochs).
    if (i % 10) == 9 and i > 0:
        PICKLE_EPOCH = \
            f'./ani1/models/ani_graphs_{instance_code}_epoch{i+1}.pickle'
        with open(PICKLE_EPOCH, "wb") as f:
            pickle.dump(model_info, f, protocol=5)

###################################################
# Load trained model & evaluate test set once again
with open(PICKLE_FILE, "rb") as f:
    model_info = pickle.load(f)

# Instantiate full model
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
    None,
    method=JatModel.calc_potential_energy
)

# Load best model parameters
model_params = flax.serialization.from_state_dict(
    template_params,
    model_info.params)
print(f"loaded best parameter state from file")

# Evaluate statistics
statistics = validation_step(model_params)
mae = statistics["force_MAE"]
rmse = statistics["force_RMSE"]
test_statistics = test_step(model_params)
test_mae = test_statistics["energy_MAE"]
test_rmse = test_statistics["energy_RMSE"]
print(
        f"Validation & Test set statistics after {N_EPOCHS+1} epochs: \n"
        f"VAL  RMSE = {rmse} {validation_units['force_RMSE']}. \n"
        f"VAL  MAE  = {mae} {validation_units['force_MAE']}. \n"
        f"TEST RMSE = {test_rmse} {validation_units['force_RMSE']}. \n"
        f"TEST MAE  = {test_mae} {validation_units['force_MAE']}.  "
)
