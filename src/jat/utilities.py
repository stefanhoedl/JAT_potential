#!/usr/bin/env python
# Copyright 2019-2022 The NeuralIL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import struct

import numpy as onp
#import scipy as sp
#import scipy.integrate
#import scipy.stats

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import jax.random

# This module contains miscellaneous utilities used throughout the code.

__all__ = [
    "draw_urandom_int32", "create_array_shuffler", "get_max_number_of_neighbors"
]


def draw_urandom_int32():
    "Generate a 32-bit random integer suitable for seeding the PRNG in JAX."
    return struct.unpack("I", os.urandom(4))[0]


def create_array_shuffler(rng):
    """Create a function able to reshuffle arrays in a consistent manner.

    Args:
        rng: A JAX splittable pseudo-random number generator, which is consumed.

    Returns:
        A function of a single argument that will return a copy of that argument
        reshuffled along the first axis as a JAX array. The result will always
        be the same for the same input. Arrays of the same length will be sorted
        correlatively.
    """
    def nruter(in_array):
        return jax.random.permutation(rng, jnp.asarray(in_array))

    return nruter

def _get_max_number_of_neighbors(coordinates, types, cutoff, cell_size=None):
    """
    Return the maximum number of neighbors within a cutoff in a configuration.

    The central atom itself is not counted.

    Parameters
    ----------
    coordinats: An (n_atoms, 3) array of atomic positions.
    cutoff: The maximum distance for two atoms to be considered neighbors.
    cell_size: Unit cell vector matrix (3x3) if the system is periodic.

    Returns
    -------
    The maximum number of atoms within a sphere of radius cutoff around another
    atom in the configuration provided.
    """
    cutoff2 = cutoff * cutoff
    delta = coordinates - coordinates[:, jnp.newaxis, :]
    if cell_size is not None:
        delta -= jnp.einsum(
            "ijk,kl",
            jnp.round(jnp.einsum("ijk,kl", delta, jnp.linalg.inv(cell_size))),
            cell_size
        )
    distances2 = jnp.sum(delta**2, axis=2)
    #n_neighbors = (distances2 < cutoff2).sum(axis=1)
    n_neighbors = jnp.logical_and(types >= 0, distances2 < cutoff2).sum(axis=1)
    return jnp.squeeze(jnp.maximum(0, n_neighbors.max() - 1))


def get_max_number_of_neighbors(coordinates, types, cutoff, cell_size=None):
    return int(
        _get_max_number_of_neighbors(coordinates, types, cutoff, cell_size=None)
    )