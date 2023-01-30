import numpy as np

from sde_hjb_solver.hjb_solver_2d_st import SolverHJB2D
from sde_hjb_solver.controlled_sde_2d import *
from sde_hjb_solver.base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the 1d HJB equation associated to' \
                         'the overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # set dimension
    d = 2

    # initialize hjb solver
    #sde = DoubleWellStoppingTime2D(beta=args.beta, alpha=np.full(2, args.alpha_i))
    sde = DoubleWellCommittor2D(beta=args.beta, alpha=np.full(2, args.alpha_i))

    # initialize hjb solver
    sol_hjb = SolverHJB2D(sde, h=args.h)

    #sde.compute_mfht()
    sde.discretize_domain_2d(sol_hjb.h)

    # compute hjb solution 
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

    sde.plot_2d_potential()

    # evaluate in grid
    sol_hjb.get_perturbed_potential_and_drift()

    sol_hjb.plot_2d_psi()
    sol_hjb.plot_2d_value_function()
    sol_hjb.plot_2d_perturbed_potential()#(ylim=(0, 20))
    sol_hjb.plot_2d_control()#(ylim=(-0.15, 20))
    #sol_hjb.plot_2d_perturbed_drift()

    #sol_hjb.plot_1d_mfht()


if __name__ == "__main__":
    main()
