from typing import Sequence

from jax import numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    """
    A simple Multilayer perceptor model.
    """

    features: Sequence[int]
    act_fn : callable
    final_act_fn : callable

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.act_fn(nn.Dense(feat)(x))
        x = self.final_act_fn(nn.Dense(self.features[-1])(x))
        return x

