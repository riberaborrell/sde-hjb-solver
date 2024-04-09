import numpy as np

from sde_hjb_solver.hjb_solver_1d_st import SolverHJB1D
from sde_hjb_solver.controlled_sde_1d import *
from sde_hjb_solver.base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes finite difference solution of the 1d HJB equation'
    return parser

def main():
    args = get_base_parser().parse_args()

    # choose sde
    if args.setting == 'mgf':
        SDE = DoubleWellMGF1D
    elif args.setting == 'committor':
        SDE = DoubleWellCommittor1D

    # initialize sde
    sde = SDE(beta=args.beta, alpha=args.alpha_i)

    # initialize hjb solver
    sol_hjb = SolverHJB1D(sde, h=args.h, load=args.load)

    # compute hjb solution 
    if not args.load:
        sol_hjb.solve_bvp()
        sol_hjb.compute_value_function()
        sol_hjb.compute_optimal_control()

        if sol_hjb.sde.setting == 'mgf':
            sol_hjb.mfht = sol_hjb.sde.compute_mfht(delta=1e-5)
        sol_hjb.save()

    # report solution
    if args.report:
        sol_hjb.write_report(x=-1.)

    # plot
    if not args.plot:
        return

    # evaluate in grid
    if sol_hjb.sde.is_overdamped_langevin:
        sol_hjb.get_perturbed_potential_and_drift()

    sol_hjb.plot_1d_psi()
    sol_hjb.plot_1d_value_function()
    sol_hjb.plot_1d_control()

    if sol_hjb.sde.is_overdamped_langevin:
        sol_hjb.plot_1d_perturbed_potential()
        sol_hjb.plot_1d_perturbed_drift()

    if hasattr(sol_hjb, 'mfht'):
        sol_hjb.plot_1d_mfht()

if __name__ == "__main__":
    main()
