import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--d',
        type=int,
        default=1,
        help='Set the dimension d. Default: 1',
    )
    parser.add_argument(
        '--theta',
        type=float,
        default=1.,
        help='Set theta parameter of Ornstein-Uhlenbeck. Default: 1.',
    )
    parser.add_argument(
        '--alpha-i',
        type=float,
        default=1.,
        help='Set barrier height of the i-th coordinate for the multidimensional extension \
              of the double well potential. Default: 1.',
    )
    parser.add_argument(
        '--alpha-j',
        type=float,
        default=1.,
        help='Set barrier height of the j-th coordinate for the multidimensional extension \
              of the double well potential. Default: 1.',
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.,
        help='Set the diffusion term parameter. Default: 1.',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='Set the inverse of the the temperature parameter. Default: 1.',
    )
    parser.add_argument(
        '--nu-i',
        type=float,
        default=3.,
        help='Set nd quadratic one well i-th parameters. Default: 1.',
    )
    parser.add_argument(
        '--nu-j',
        type=float,
        default=3.,
        help='Set nd quadratic one well j-th parameters. Default: 1.',
    )
    parser.add_argument(
        '--h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.005,
        help='Set the time discretization increment for the hjb sol with det time horizont. Default: 0.005',
    )
    parser.add_argument(
        '--T',
        type=float,
        default=1.,
        help='Set deterministic time horizont. Default: 1.',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Write / Print report. Default: False',
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load already computed hjb results. Default: False',
    )
    return parser
