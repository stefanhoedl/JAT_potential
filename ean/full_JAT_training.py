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

if os.getcwd().startswith('/workspace/'):
    sys.path.append('/workspace/JAT_potential/src')
    log_wandb = True
else:
    sys.path.append('/home/stefan/tu/JAT_potential/src')
    log_wandb = True

from jat.jat_model import JatCore, JatModel, GraphGenerator, JATModelInfo
from jat.training import * 
from jat.utilities import create_array_shuffler, draw_urandom_int32, \
    get_max_number_of_neighbors

## Training Config
TRAINING_FRACTION = .865
N_BATCH = 8
N_EVAL_BATCH = 32
SEED = 421
LOG_COSH_PARAMETER = 1e0  # In angstrom / eV
LR_MIN, LR_MAX, LR_END = 0.5e-4, 0.3e-3, 0.5e-6
N_EPOCHS = 2001
N_PAIR = 15

## JAT MODEL
LAYER_DIMS = [48, 48, 48, 48, 48, 48, 48]
GRAPH_CUT = 5
EMBED_D = 48
N_HEADS = 1

instance_code = draw_urandom_int32()
CONFIGS_DFT = "./configurations.json" 
PICKLE_FILE = f"./ean/models/JAT_EAN15_{instance_code}.pickle"

type_cation = ["N", "H", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
type_anion = ["N", "O", "O", "O"]
types = N_PAIR * type_cation + N_PAIR * type_anion

sorted_elements = sorted(set(types))
type_dict = OrderedDict()
for i, k in enumerate(sorted_elements):
    type_dict[k] = i
types = jnp.array([type_dict[i] for i in types])
n_atoms = len(types)

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

n_configurations = len(positions)
types = jnp.array([types for i in range(n_configurations)])

n_train = int(TRAINING_FRACTION * n_configurations)
n_validate = (n_configurations - n_train) //2
n_test = n_configurations - n_train - n_validate
print(f"\t- {n_train} will be used for training")
print(f"\t- {n_validate} will be used for validation")
print(f"\t- {n_test} will be used for test")
n_types = int(types.max()) + 1

rng = jax.random.PRNGKey(SEED)
rng, shuffler_rng = jax.random.split(rng)
shuffle = create_array_shuffler(shuffler_rng)
cells = shuffle(cells)
positions = shuffle(positions)
energies = shuffle(energies)
types = shuffle(types)
forces = shuffle(forces)

def split_array(in_array):
    "Split an array in training and validation sections."
    return jnp.split(in_array, (n_train, n_train + n_validate, 
        n_train + n_validate + n_test))[:3]

cells_train, cells_validate, cells_test = split_array(cells)
positions_train, positions_validate, positions_test = split_array(positions)
types_train, types_validate, types_test = split_array(types)
energies_train, energies_validate, energies_test = split_array(energies)
forces_train, forces_validate, forces_test = split_array(forces)

graph_neighbors = 1
for p, t, c in zip(positions, types, cells):
    graph_neighbors = max(
        graph_neighbors,
        get_max_number_of_neighbors(
            jnp.asarray(p), 
            jnp.asarray(t), 
            GRAPH_CUT, 
            jnp.asarray(c)
        )
    )
print(f"Maximum of {graph_neighbors} neighbors considered for Graph generation")

core_model = JatCore(layer_dims = LAYER_DIMS, 
    n_head = N_HEADS)
graph_gen = GraphGenerator(n_atoms, 
    GRAPH_CUT, 
    cells_train[0],
    graph_neighbors)
dynamics_model = JatModel(
    n_types, EMBED_D, graph_gen, core_model
)

# Create the minimizer.
optimizer = create_one_cycle_minimizer(
    n_train // N_BATCH, LR_MIN, LR_MAX, LR_END
)

rng, init_rng = jax.random.split(rng)
model_params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=JatModel.calc_forces
)

optimizer_state = optimizer.init(model_params)

# Create the function that will compute the contribution to the loss from a
# single data point. In this case, our loss will not make use of the energies.
log_cosh = create_log_cosh(LOG_COSH_PARAMETER)

if log_wandb:
    import wandb
    wandb.init(project='il-jat-EAN-production', config={
    "N_EPOCHS" : N_EPOCHS,
    "N_PAIR":N_PAIR,
    "TRAINING_FRACTION" : TRAINING_FRACTION,
    "GRAPH_CUT": GRAPH_CUT,
    "PICKLE_FILE" : PICKLE_FILE,
    "N_BATCH": N_BATCH,
    "LOG_COSH_PARAMETER": LOG_COSH_PARAMETER,
    "SEED": SEED,
    "layer_dims": LAYER_DIMS,
    "N_HEADS": N_HEADS,
    "LR_MIN":LR_MIN,
    "LR_MAX": LR_MAX,
    "LR_END":LR_END,
    "instance": instance_code,
    })
    config = wandb.config

# Get flattened key-value list of trainable parameters.
flat_params = {'/'.join(k[-2:]): v.shape for k, v in \
    flax.traverse_util.flatten_dict(flax.core.unfreeze(model_params)).items()}
print(flat_params)

def calc_loss_contribution(pred_energy, pred_forces, obs_energy, obs_forces):
    "Return the log-cosh of the difference between predicted and actual forces."
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
    log_wandb = True
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
    # Reset the training schedule.
    optimizer_state = reset_one_cycle_minimizer(optimizer_state)
    # Run a full epoch.
    optimizer_state, model_params = training_epoch(
        optimizer_state, model_params
    )
    # Evaluate the results.
    statistics = validation_step(model_params)
    mae = statistics["force_MAE"]
    rmse = statistics["force_RMSE"]
    # Print the relevant statistics.

    if log_wandb:
        wandb.log({"rmse": rmse.copy(), "mae": mae.copy()}, commit=False)
        
    # Save the state only if the validation MAE is minimal.
    if mae < min_mae:
        model_info = JATModelInfo(
            model_name="IL_JAT",
            model_details=f"EAN {N_PAIR}",
            timestamp=datetime.now(),
            graph_cut=GRAPH_CUT,
            graph_neighbors=graph_neighbors,
            sorted_elements=sorted_elements,
            embed_d=EMBED_D,
            layer_dims=LAYER_DIMS,
            n_head=N_HEADS,
            n_atoms = N_PAIR * 15,
            constructor_kwargs={"cell_size": cells_train[0]},
            random_seed=SEED,
            params=flax.serialization.to_state_dict(model_params),
            specific_info=None
        )
        with open(PICKLE_FILE, "wb") as f:
            #print("- Saving the most recent state")
            pickle.dump(model_info, f, protocol=5)
        print(f"woooo {mae} mae & {rmse} rmse")
        min_mae = mae
    
    # Periodically save the best model.
    if i % 250 == 0 and i>0:
        PICKLE_EPOCH_FILE = f"./ean/models/JAT_EAN15_{instance_code}_epoch{i+1}.pickle"
        with open(PICKLE_EPOCH_FILE, "wb") as f:
                pickle.dump(model_info, f, protocol=5)
    

    test_statistics = test_step(model_params)
    test_mae = test_statistics["force_MAE"]
    test_rmse = test_statistics["force_RMSE"]
    print(
        f"VALIDATION: \t"
        f"RMSE = {rmse} {validation_units['force_RMSE']}. "
        f"MAE = {mae} {validation_units['force_MAE']}."
    )
    print(
        f"TEST: \t\t"
        f"RMSE = {test_rmse} {validation_units['force_RMSE']}. "
        f"MAE = {test_mae} {validation_units['force_MAE']}."
    )
    if log_wandb:
        wandb.log({"test_rmse": test_rmse.copy(), "test_mae": test_mae.copy()})

# load params to verify
with open(PICKLE_FILE, "rb") as f:
    model_info = pickle.load(f)

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

template_params = dynamics_model.init(
    init_rng,
    jnp.zeros((model_info.n_atoms, 3), dtype=jnp.float32),
    jnp.zeros(model_info.n_atoms, dtype=jnp.int32),
    jnp.eye(3),
    method=JatModel.calc_forces
)
model_params = flax.serialization.from_state_dict(template_params, 
    model_info.params)

statistics = validation_step(model_params)
mae = statistics["force_MAE"]
rmse = statistics["force_RMSE"]
test_statistics = test_step(model_params)
test_mae = test_statistics["force_MAE"]
test_rmse = test_statistics["force_RMSE"]
print(
        f"Validation set statistics with loaded parameters: \n"
        f"RMSE = {rmse} {validation_units['force_RMSE']}. "
        f"MAE = {mae} {validation_units['force_MAE']}."
    )
print(
        f"Test set statistics with loaded parameters: \n"
        f"RMSE = {test_rmse} {validation_units['force_RMSE']}. "
        f"MAE = {test_mae} {validation_units['force_MAE']}."
    )