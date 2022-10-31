# Copyright 2022 Stefan Hödl
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

from setuptools import setup, find_packages

setup(
    name="JAT",
    version="1.0",
    description="Jraph Attention Networks: an attention-based atomistic deep learning potential",
    author="Stefan Hödl",
    author_email="stefano.hoedl@gmail.com",
    python_requires=">=3.8",
    packages=find_packages(where="./src/jat"),
    install_requires=[
        "jax >= 0.3.10",
        "jaxlib >= 0.3.10",
        "jraph >= 0.0.5.dev0",
        "wandb >= 0.13.0",
        "flax >= 0.4.2",
        "optax >= 0.1.2",
        "ase >= 3.22.0"
        "numpy >= 1.19.5",
        "tqdm >= 4.61.1",
        "h5py >= 3.1.0",
    ]
)

# jaxlib + cuda:
# https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.10+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl
# jax                               0.3.10
# jaxlib                            0.3.10+cuda11.cudnn805