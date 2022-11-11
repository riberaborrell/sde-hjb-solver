import numpy as np

from hjb_solver_1d import SolverHJB1D
from controlled_sde_1d import *
from base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the 1d HJB equation associated to' \
                         'the overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # set dimension
    d = 1

    # initialize hjb solver
    sde = DoubleWellStoppingTime1D(beta=args.beta, alpha=args.alpha_i)
    #sde = DoubleWellCommittor1D(beta=args.beta, alpha=args.alpha_i)
    #sde = SkewDoubleWellStoppingTime1D(beta=args.beta)
    #sde = BrownianMotionCommittor1D()

    # initialize hjb solver
    sol_hjb = SolverHJB1D(sde, h=args.h)

    #sde.compute_mfht()

    # compute hjb solution 
    sol_hjb.start_timer()
    sol_hjb.solve_bvp()
    sol_hjb.compute_value_function()
    sol_hjb.compute_optimal_control()
    #sol_hjb.compute_mfht()

    # report solution
    if args.report:
        sol_hjb.write_report(x=-1.)

    # plot
    if not args.plot:
        return

    # evaluate in grid
    sol_hjb.get_controlled_potential_and_drift()

    sol_hjb.plot_1d_psi()
    sol_hjb.plot_1d_value_function()
    sol_hjb.plot_1d_controlled_potential()#(ylim=(0, 20))
    sol_hjb.plot_1d_control()#(ylim=(-0.15, 20))
    sol_hjb.plot_1d_controlled_drift()

    #sol_hjb.plot_1d_mfht()


if __name__ == "__main__":
    main()
