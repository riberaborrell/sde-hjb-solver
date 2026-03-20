import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(
        description=None,
        usage="%(prog)s [--foo FOO] [--bar BAR]",
    )
    problem = parser.add_argument_group("problem options")
    problem.add_argument(
        '--setting',
        metavar='SETTING',
        choices=['mgf', 'committor'],
        default='mgf',
        help='Set setting type: %(choices)s (default: %(default)s)',
    )
    problem.add_argument(
        '--problem',
        metavar='PROBLEM',
        choices=['brownian', 'doublewell', 'skew', 'triplewell', 'ryckbell', 'fivewell', 'mueller'],
        default='doublewell',
        help='Set problem type (for overdamped langevin dynamics set the type of potential): %(choices)s (default: %(default)s)',
    )
    problem.add_argument(
        '--d',
        metavar='DIMENSION',
        type=int,
        default=1,
        help='Set the dimension d (default: %(default)s)',
    )
    problem.add_argument(
        '--theta',
        metavar='VAL',
        type=float,
        default=1.,
        help='Set theta parameter of Ornstein-Uhlenbeck (default: %(default)s)',
    )
    problem.add_argument(
        '--alpha',
        metavar='VAL',
        type=float,
        nargs='+',
        default=[1.],
        help='Set barrier height parameter of the given potential. Is a vector of size d (provide one value per dimension, e.g. --alpha 1.0 2.0 for d = 2) (default: %(default)s)',
    )
    problem.add_argument(
        '--sigma',
        type=float,
        default=1.,
        help='Set the diffusion term parameter (default: %(default)s)',
    )
    problem.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='Set the inverse of the temperature parameter (default: %(default)s)',
    )
    problem.add_argument(
        '--nu-i',
        type=float,
        default=3.,
        help='Set nd quadratic one well i-th parameters (default: %(default)s)',
    )
    problem.add_argument(
        '--T',
        metavar='TIME_HORIZONT',
        type=float,
        default=1.,
        help='Set deterministic time horizont (default: %(default)s)',
    )
    algorithm = parser.add_argument_group("algorithm options")
    algorithm.add_argument(
        '--seed',
        type=int,
        help='Set the seed for RandomState (default: %(default)s)',
    )
    algorithm.add_argument(
        '--h',
        metavar='SPACE_DISCRETIZATION',
        type=float,
        default=0.1,
        help='Set the discretization step size for each dimension (default: %(default)s)',
    )
    algorithm.add_argument(
        '--dt',
        metavar='TIME_DISCRETIZATION',
        type=float,
        default=0.005,
        help='Set the time discretization increment for the finite time horizont setting (default: %(default)s)',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Display plots (default: %(default)s)',
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Log report (default: %(default)s)',
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load already computed results (default: %(default)s)',
    )
    return parser
