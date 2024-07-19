import numpy as np

from sde_hjb_solver.hjb_solver_2d_st import SolverHJB2D
from sde_hjb_solver.controlled_sde_2d import *
from sde_hjb_solver.base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes finite difference solution of the 2d HJB equation'
    return parser

def main():
    args = get_parser().parse_args()

    # choose sde
    if args.problem == 'brownian' and args.setting == 'mgf':
        SDE = BrownianMotionMgf2D
    elif args.problem == 'brownian' and args.setting == 'committor':
        SDE = BrownianMotionCommittor2D
    elif args.problem == 'doublewell' and args.setting == 'mgf':
        SDE = DoubleWellMgf2D
    elif args.problem == 'doublewell' and args.setting == 'committor':
        SDE = DoubleWellCommittor2D
    elif args.problem == 'triplewell' and args.setting == 'mgf':
        SDE = TripleWellMgf2D
    elif args.problem == 'triplewell' and args.setting == 'committor':
        SDE = TripleWellCommittor2D
    elif args.problem == 'mueller' and args.setting == 'mgf':
        SDE = MuellerBrownMgf2D
    elif args.problem == 'mueller' and args.setting == 'committor':
        SDE = MuellerBrownCommittor2D
    else:
        raise NotImplementedError


    # initialize sde
    sde = SDE(
        beta=args.beta,
        alpha=np.array(args.alpha),
        ts_pot_level=0.25,
    )

    # initialize hjb solver
    sol_hjb = SolverHJB2D(sde, h=args.h, load=args.load)

    # compute hjb solution 
    if not args.load:
        sol_hjb.solve_bvp()
        sol_hjb.compute_value_function()
        sol_hjb.compute_optimal_control()

        if sol_hjb.sde.setting == 'mgf':
            sol_hjb.mfht = sol_hjb.sde.compute_mfht()

        sol_hjb.save()

    # report solution
    if args.report:
        sol_hjb.write_report(x=-1.)

    # plot
    if not args.plot:
        return


    # evaluate in grid
    if sol_hjb.sde.is_overdamped_langevin:
        sde.plot_2d_potential()
        sol_hjb.get_perturbed_potential_and_drift()

    sol_hjb.plot_2d_psi()
    sol_hjb.plot_2d_value_function()
    sol_hjb.plot_2d_control()

    if sol_hjb.sde.is_overdamped_langevin:
        sol_hjb.plot_2d_perturbed_potential()
        #sol_hjb.plot_2d_perturbed_drift()

    if hasattr(sol_hjb, 'mfht'):
        sol_hjb.plot_2d_mfht()

if __name__ == "__main__":
    main()
