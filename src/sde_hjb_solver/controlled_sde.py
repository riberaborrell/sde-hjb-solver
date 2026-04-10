import functools
from typing import Any, Optional, Union

import numpy as np

from sde_hjb_solver.functions import constant, quadratic_one_well

class ControlledSDE:
    """Base class for controlled stochastic differential equations.

    This class defines common configuration for different settings
    (e.g., MGF, committor, finite time horizon). Concrete SDEs are implemented
    in dimension-specific subclasses and should provide target-set logic and
    model-specific parameters.

    Attributes:
        d: Dimension of the state space.
        domain: Domain bounds. For 1D, typically (lb, ub). For 2D, typically
            an array of shape (d, 2) with lower/upper bounds per axis.
        setting: Active problem setting identifier (e.g., "mgf", "committor").
        f: Running cost function.
        g: Terminal cost function.
    """

    def __init__(
        self,
        d: int,
        domain: Optional[Union[tuple, np.ndarray]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a controlled SDE container.

        Args:
            d: Dimension of the state space.
            domain: Domain bounds for the state space.
            **kwargs: Additional parameters passed by subclasses (unused here).
        """

        # dimension
        self.d = d

        # domain bounds
        self.domain = domain

    def set_mgf_setting(self, lam: float = 1.0) -> None:
        """Set the moment generating function (MGF) of the first hitting time setting.

        Args:
            lam: MGF parameter.
        """
        # set mgf problem flag
        self.setting = 'mgf'

        # running and final costs
        self.lam = lam
        self.f = functools.partial(constant, a=lam)
        self.g = functools.partial(constant, a=0.)

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_mgf

    def set_committor_setting(self, epsilon: float = 1e-10) -> None:
        """Set the committor probability setting.

        Args:
            epsilon: Small positive constant used to regularize log terms.
        """
        # set committor problem flag
        self.setting = 'committor'

        # running and final costs
        self.epsilon = epsilon
        self.f = lambda x: 0
        self.g = lambda x: np.where(
            self.is_target_set_b(x),
            -np.log(1+epsilon),
            -np.log(epsilon),
        )

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_committor

    def set_finite_time_horizon_setting(self, nu: float = 1.0) -> None:
        """Set the finite time horizon setting.

        Args:
            nu: parameter scaling the quadratic one well potential.
        """
        # set committor problem flag
        self.setting = 'finite_time_horizon'

        # running and final costs
        self.nu = nu
        self.f = lambda x: 0
        self.g = functools.partial(quadratic_one_well, nu=nu)

    def set_fht_probs_setting(self, T: float = 1.0, epsilon: float = 1e-10) -> None:
        """Set the first hitting time probabilities setting.

        Args:
            T: Time horizon used by the FHT probabilities formulation.
            epsilon: Small positive constant used to regularize log terms.
        """
        # set committor problem flag
        self.setting = 'fht_probabilities'

        # finite time horizon
        self.T = T

        # running and final costs
        self.epsilon = epsilon
        self.f = lambda x: 0
        self.g = lambda x: np.where(
            self.is_target_set(x),
            -np.log(1+epsilon),
            -np.log(epsilon),
        )



    def __str__(self) -> str:
        """Return a short identifier for output paths and logging."""
        return f'{self.name}__{self.params_str}'
