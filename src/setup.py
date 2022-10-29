# Copyright 2019-2021 The NeuralIL contributors
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

from distutils.core import setup
from setuptools.command.install import install

setup(
    name="JAT",
    version="1.0",
    description="Jraph Attention Networks: an attention-based atomistic deep learning potential",
    author="Stefan Hödl",
    author_email="stefano.hoedl@gmail.com",
    packages=[
        "jat"
    ],
    install_requires=[
        "jax",
        "jaxlib",
        "jraph",
        "wandb",
        "flax",
        "optax",
        "numpy",
        "tqdm",
        "h5py",
        "ase"
    ]
)

# jaxlib + cuda:
# https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.10+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl

# jax                               0.3.10
# jaxlib                            0.3.10+cuda11.cudnn805
# flax                              0.4.2
# optax                             0.1.2
# ase                               3.22.0
# jraph                             0.0.5.dev0
# wandb                             0.13.0
# networkx                          2.5.1