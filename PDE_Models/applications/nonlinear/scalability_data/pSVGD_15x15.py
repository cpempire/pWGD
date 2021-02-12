from model_lognormal_15x15 import *

import time


# check the stein/options to see all possible choices
options["type_optimization"] = "gradientDescent"
options["is_projection"] = True
options["tol_projection"] = 1.e-2
options["type_projection"] = "fisher"
options["is_precondition"] = False
options["type_approximation"] = "fisher"
options["coefficient_dimension"] = 256
options["add_dimension"] = 0

options["number_particles"] = 64
options["number_particles_add"] = 0
options["add_number"] = 0
options["add_step"] = 5
options["add_rule"] = 1

options["type_scaling"] = 1
options["type_metric"] = "posterior_average"  # posterior_average

options['WGF'] = False

options["type_Hessian"] = "lumped"
options["low_rank_Hessian"] = False
options["rank_Hessian"] = 256
options["rank_Hessian_tol"] = 1.e-2
options["low_rank_Hessian_average"] = False
options["rank_Hessian_average"] = 256
options["rank_Hessian_average_tol"] = 1.e-2
options["gauss_newton_approx"] = True  # if error of unable to solve linear system occurs, use True

options["max_iter"] = 200
options["step_tolerance"] = 1e-7
options["step_projection_tolerance"] = 1e-3
options["line_search"] = True
options["search_size"] = 1e-1
options["max_backtracking_iter"] = 10
options["cg_coarse_tolerance"] = 0.5e-2
options["print_level"] = -1
options["save_number"] = 20
options["plot"] = True

# generate particles
particle = Particle(model, options, comm)

# evaluate the variation (gradient, Hessian) of the negative log likelihood function at given particles
variation = Variation(model, particle, options, comm)

# evaluate the kernel and its gradient at given particles
kernel = Kernel(model, particle, variation, options, comm)

t0 = time.time()

solver = GradientDescent(model, particle, variation, kernel, options, comm)

solver.solve()

print("GradientDecent solving time = ", time.time() - t0)
