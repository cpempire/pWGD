
from __future__ import absolute_import, division, print_function

import time
import matplotlib.pyplot as plt
# from integration import Euler, RK2, RK4
from scipy.integrate import odeint

import autograd.numpy as np
from autograd.numpy import multiply as ewm
from autograd import grad, hessian, hessian_vector_product
from autograd.misc import flatten

from scipy.optimize import minimize, NonlinearConstraint, BFGS, Bounds

import sys
sys.path.append("../../")
from utils.integration import Euler, RK2, RK4

from initial_input import *
from initial_model import Model
import pickle

first_confirmed = np.where(states[fips]["positive"] > 100)[0][0]
data_confirmed = states[fips]["positive"][first_confirmed:]
number_days_data = len(data_confirmed)

first_hospitalized = np.where(states[fips]["hospitalizedCurrently"] > 10)[0][0]
lag_hospitalized = first_hospitalized - first_confirmed
data_hospitalized = states[fips]["hospitalizedCurrently"][first_hospitalized:]

first_deceased = np.where(states[fips]["death"] > 10)[0][0]
lag_deceased = first_deceased - first_confirmed
data_deceased = states[fips]["death"][first_deceased:]


def initialization(configurations, parameters, controls):
    # initialize the optimization problem with controls terminate at the end of observation (today)
    
    y0, t_total, N_total, number_group, population_proportion, \
    t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

    alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = controls

    model = Model(configurations, parameters, controls)

    t0 = time.time()
    solution = RK2(model.seir, y0, t_total, parameters, controls)
    # np.savez(savefilename, t=t, solution=solution, controls=controls)
    print("solve time by RK2 method", time.time() - t0)

    solution_group = model.grouping(solution)

    print("# total infected = ", N_total - solution_group[-1, 0],
          "# total death = ", solution_group[-1, 7],
          "maximum # hospitalized = ", np.max(solution_group[:, 5]))

    simulation_first_confirmed = np.where(solution_group[:, 8] >= data_confirmed[0])[0][0]
    # simulation_first_confirmed = np.where(solution_group[:, 8] > data_confirmed[0]-1)[0][0]
    # simulation_first_confirmed = 40
    print("day for first 100 confirmed = ", simulation_first_confirmed)

    # control frequency and time
    # number_days_per_control_change = 7
    number_control_change_times = np.floor_divide((simulation_first_confirmed + len(data_confirmed)),
                                                  number_days_per_control_change)
    number_control_days = number_days_per_control_change * number_control_change_times

    t_control = np.linspace(0, number_control_days, number_control_change_times)

    number_days = number_control_days
    t_total = np.linspace(0, number_days, 1 * number_days + 1)

    # t_control = np.linspace(25, 25 + (number_control_change_times - 1) * number_days_per_control_change,
    #                         number_control_change_times)

    alpha = np.array([0.0 for i in range(number_control_change_times)])  # S to E social distancing
    # q = np.array([0.0 for i in range(number_control_change_times)])  # S to Q quarantine
    # tau = np.array([0.1 for i in range(number_control_change_times)])  # E to I confirmation ratio
    # HFR = np.array([0.2 for i in range(number_control_change_times)])
    # kappa = np.array([0.04 for i in range(number_control_change_times)])

    controls = (alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q)
    # controls = alpha

    configurations = (y0, t_total, N_total, number_group, population_proportion,
                      t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls)

    return simulation_first_confirmed, configurations, parameters, controls


# define the penalization function for the control variables
def penalization(controls, coeffs, kappas):

    result = 0.
    # # penalization on alpha, q, tau from decreasing
    # increasing = [0.01, 1., 1.]
    # for i in range(1):
    #     result = result + increasing[i]*np.sum(np.mean(ewm(coeffs[i][1:], np.power(np.abs(np.diff(controls[i], axis=0))-np.diff(controls[i], axis=0), 1.)), axis=0))
    result = np.sum(np.mean(ewm(coeffs[0][1:], np.power(np.abs(np.diff(controls[0], axis=0))-np.diff(controls[0], axis=0), 1.)), axis=0))
    # result = result + 1.*np.sum(np.mean(ewm(coeffs[3][1:], np.power(np.abs(np.diff(controls[3], axis=0))+np.diff(controls[3], axis=0), 1.)), axis=0))

    # # # penalization on HFR, beta, delta, kappa, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q from mean
    # for j in range(2, len(parameters)):
    #     result = result + 0.5/500 * np.sum(np.power(np.divide(controls[j+3] - parameters[j], parameters[j]/2), 2))

    return result


# define the objective function for deaths, hospitalized, and control
def objective(flat_args):
    # objective uses flattend arguments to facilitate autograd functionality

    # unflatten, parameters, weights = args

    controls = unflatten(flat_args)
    solution = RK2(model.seir, y0, t_total, parameters, controls)
    solution = model.grouping(solution)

    ws, coeffs, kappas = weights

    day = simulation_first_confirmed
    simulation_confirmed = solution[day:day + len(data_confirmed), 8]
    # penalty_confirmed = np.sum(np.power(np.log(simulation_confirmed) - np.log(data_confirmed), 2))
    penalty_confirmed = np.sum(np.power(np.log(np.diff(simulation_confirmed)) - np.log(np.diff(data_confirmed)), 2))
    # penalty_confirmed = np.sum(np.log(1+np.power(simulation_confirmed - data_confirmed, 2)))
    # penalty_confirmed = np.sum(np.log(1+np.abs(simulation_confirmed - data_confirmed)))
    # penalty_confirmed = np.sum(np.power(simulation_confirmed - data_confirmed, 2)/data_confirmed[-1]**2)

    day = simulation_first_confirmed + lag_hospitalized
    simulation_hospitalized = solution[day:day + len(data_hospitalized), 5]
    penalty_hospitalized = np.sum(np.power(np.log(simulation_hospitalized) - np.log(data_hospitalized), 2))
    # penalty_hospitalized = np.sum(np.log(1+np.power(simulation_hospitalized - data_hospitalized, 2)))
    # penalty_hospitalized = np.sum(np.log(1+np.abs(simulation_hospitalized - data_hospitalized)))
    # penalty_hospitalized = np.sum(np.power(simulation_hospitalized - data_hospitalized, 2)/data_hospitalized[-1]**2)

    day = simulation_first_confirmed + lag_deceased
    simulation_deceased = solution[day:day + len(data_deceased), 7]
    penalty_deceased = np.sum(np.power(np.log(simulation_deceased) - np.log(data_deceased), 2))
    penalty_deceased = penalty_deceased + np.sum(np.power(np.log(np.diff(simulation_deceased)) - np.log(np.diff(data_deceased)), 2))
    # penalty_deceased = np.sum(np.log(1+np.power(simulation_deceased - data_deceased, 2)))
    # penalty_deceased = np.sum(np.log(1+np.abs(simulation_deceased - data_deceased)))
    # penalty_deceased = np.sum(np.power(simulation_deceased - data_deceased, 2)/data_deceased[-1]**2)

    # if number_group == 1:
    #     # simulation_first_confirmed = np.where(solution[:, 8] > 100)[0][0]
    #
    #     day = simulation_first_confirmed
    #     simulation_confirmed = solution[day:day + len(data_confirmed), 8]
    #     penalty_confirmed = np.sum(np.power(np.log(simulation_confirmed) - np.log(data_confirmed), 2))
    #
    #     day = simulation_first_confirmed + lag_hospitalized
    #     simulation_hospitalized = solution[day:day + len(data_hospitalized), 5]
    #     penalty_hospitalized = np.sum(np.power(np.log(simulation_hospitalized) - np.log(data_hospitalized), 2))
    #
    #     day = simulation_first_confirmed + lag_deceased
    #     simulation_deceased = solution[day:day + len(data_deceased), 7]
    #     penalty_deceased = np.sum(np.power(np.log(simulation_deceased) - np.log(data_deceased), 2))
    #
    #     # infected = solution[day-20:day-20+len(data_deceased), 8] + solution[day-20:day-20+len(data_deceased), 9]
    #     #
    #     # penalty_deceased = penalty_deceased + np.sum(np.power(np.log(infected) - np.log(data_deceased) - np.log(100/0.65), 2))
    #
    # else:
    #     day = simulation_first_confirmed
    #     simulation_confirmed = np.sum(solution[day:day + len(data_confirmed), 8*number_group:9*number_group], axis=1)
    #     penalty_confirmed = np.sum(np.power(np.log(simulation_confirmed) - np.log(data_confirmed), 2))
    #
    #     day = simulation_first_confirmed + lag_hospitalized
    #     simulation_hospitalized = np.sum(solution[day:day + len(data_hospitalized), 5*number_group:6*number_group], axis=1)
    #     penalty_hospitalized = np.sum(np.power(np.log(simulation_hospitalized) - np.log(data_hospitalized), 2))
    #
    #     day = simulation_first_confirmed + lag_deceased
    #     simulation_deceased = np.sum(solution[day:day + len(data_deceased), 7*number_group:8*number_group], axis=1)
    #     penalty_deceased = np.sum(np.power(np.log(simulation_deceased) - np.log(data_deceased), 2))
    #
    #     # death = solution[day + len(data_deceased), 7*number_group:8*number_group]
    #     # simulation_death_by_age = death[:9] + death[9:]
    #     # simulation_death_by_age = simulation_death_by_age / np.sum(simulation_death_by_age)
    #     # penalty_deceased_distribution = 100. * np.sum((simulation_death_by_age - death_by_age) ** 2)
    #     # penalty_deceased = penalty_deceased + penalty_deceased_distribution
    #
    #     # simulation_deaths = np.sum(solution[ten_death_day:ten_death_day + len(deaths), 7*number_group:8*number_group], axis=1)
    #     # penalty_deceased = np.sum(np.power(np.log(simulation_deaths) - np.log(deaths), 2))
    #     #
    #     # simulation_hospitalized = np.sum(solution[ten_death_day + lag:ten_death_day + lag + len(hospitalized), 5*number_group:6*number_group], axis=1)
    #     # penalty_hospitalized = np.sum(np.power(np.log(simulation_hospitalized) - np.log(hospitalized), 2))
    #     #
    #     # simulation_positive = np.sum(solution[ten_death_day:ten_death_day + len(deaths), 8*number_group:9*number_group], axis=1)
    #     # penalty_positive = np.sum(np.power(np.log(simulation_positive) - np.log(positive), 2))

    # penalization for control
    penalty_control = penalization(controls, coeffs, kappas)

    result = ws[0]*penalty_deceased + ws[1]*penalty_hospitalized + ws[2]*penalty_confirmed + ws[3]*penalty_control

    # print("alpha, q, g_I, g_A = ", controls)
    print("weights = ", ws,
          "deceased term = ", ws[0]*penalty_deceased,
          "hospitalized term = ", ws[1]*penalty_hospitalized,
          "confirmed term = ", ws[2]*penalty_confirmed,
          "control term = ", ws[3]*penalty_control)
        # "deceased distribution = ", penalty_deceased_distribution

    return result


# define constraint function dpending on the parmaeters
def constraint(flat_args):
    # objective uses flattend arguments to facilitate autograd functionality

    controls = unflatten(flat_args)
    solution = RK2(model.model.seir, y0, t_total, parameters, controls)
    solution = model.grouping(solution)

    result = np.max(solution[:, 5]) * (1.E6 / N_total)

    # if number_group == 1:
    #     # maximum number of hospitalized cases
    #     result = np.max(solution[:, 5]) * (1.E6 / N_total)
    # else:
    #     # maximum number of hospitalized cases
    #     result = np.max(np.sum(solution[:, 5*number_group:6*number_group], axis=1)) * (1.E6/N_total)

    print("hospitalized cases per million = ", result)

    return result


def check_gradient(flat_args):
    t0 = time.time()
    objective_grad = grad(objective)
    gradient_AD = objective_grad(flat_args)
    print("time to compute gradient by autograd = ", time.time() - t0)
    # print("gradient by autograd = ", gradient_AD)

    # check gradient by finite difference
    t0 = time.time()
    gradient_FD = []
    for i in range(len(flat_args)):
        p = flat_args.copy()
        p[i] = flat_args[i] * 0.999
        obj1 = objective(p)

        p = flat_args.copy()
        p[i] = flat_args[i] * 1.001
        obj2 = objective(p)

        g = (obj2 - obj1) / (2 * 0.001 * flat_args[i])
        gradient_FD.append(g)
    print("time to compute gradient by finite difference = ", time.time() - t0)
    # print("gradient by finite difference = ", gradient_FD)

    print("relative error of gradient by FD and AD = ",
          np.linalg.norm(gradient_FD - gradient_AD) / np.linalg.norm(gradient_AD))


def check_hessian(flat_args):

    t0 = time.time()
    objective_grad = grad(objective)
    objective_hess = hessian_vector_product(objective)
    hvp_AD = objective_hess(flat_args, flat_args)
    print("time to compute hessian vector product by autograd = ", time.time() - t0)
    # print("hessian vector product by autograd = ", hvp_AD)

    # check hessian vector product by finite difference
    t0 = time.time()
    p1 = np.copy(flat_args)
    p1 = p1 - flat_args*0.001
    p2 = np.copy(flat_args)
    p2 = p2 + flat_args*0.001

    grad1 = objective_grad(p1)
    grad2 = objective_grad(p2)

    hvp_FD = (grad2 - grad1)/(2*0.001)
    print("time to compute hessian vector product by finite difference = ", time.time() - t0)
    # print("hessian vector product by finite difference = ", hvp_FD)

    print("relative error of hessian vector product by FD and AD = ", np.linalg.norm(hvp_AD-hvp_FD)/np.linalg.norm(hvp_AD))


if __name__ == "__main__":

    simulation_first_confirmed, configurations, parameters, controls = initialization(configurations, parameters, controls)

    y0, t_total, N_total, number_group, population_proportion, \
    t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

    model = Model(configurations, parameters, controls)

    solution = RK2(model.seir, y0, t_total, parameters, controls)

    Rt = model.reproduction(t_total, parameters, controls, solution)

    # l-bfgs-b optimization
    # ws = [1., 1., 100.]
    # penalty parameters for deceased, hospitalized, confirmed, and control
    ws = [2., 0., 0., 100.]

    # penalty adjusted by population in each group
    # coeffs = [np.array(
    #               [10. * population_proportion * (((i + 1) / number_control_change_times)+0.5) for i in range(number_control_change_times)]),
    #           np.array(
    #               [1. * population_proportion * (((i + 1) / number_control_change_times)+0.5) for i in range(number_control_change_times)]),
    #           np.array(
    #               [1. * population_proportion * (((i + 1) / number_control_change_times)+0.5) for i in range(number_control_change_times)]),
    #           np.array(
    #               [2. * population_proportion * (((i + 1) / number_control_change_times)+0.5) for i in range(number_control_change_times)])
    #           ]

    # coeffs = [np.array(
    #     [1. * population_proportion for i in range(number_control_change_times)]),
    #     np.array(
    #         [1. * population_proportion for i in range(number_control_change_times)]),
    #     np.array(
    #         [1. * population_proportion for i in range(number_control_change_times)]),
    #     np.array(
    #         [1. * population_proportion for i in range(number_control_change_times)])
    # ]

    coeffs = [np.array(
        [1. for i in range(number_control_change_times)]),
        np.array(
            [1. for i in range(number_control_change_times)]),
        np.array(
            [1. for i in range(number_control_change_times)]),
        np.array(
            [1. for i in range(number_control_change_times)])
    ]

    # kappas = [2., 2., 2.]
    kappas = [1., 1., 1.]

    weights = (ws, coeffs, kappas)

    flat_args, unflatten = flatten(controls)

    args = (unflatten, parameters, weights)

    # check_gradient(flat_args)

    # check_hessian(flat_args)

    # ################################# low and upper bounds for the control variables
    ones = np.ones(number_group)

    # low and upper bound for
    # [alpha, q, tau, HFR, beta, delta, kappa, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q]
    # alpha, q, tau are time dependent
    # HFR, beta, delta, kappa, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = parameters

    # lower_bound = [0.1 * ones, 0.0 * ones, 0.02 * ones]
    # upper_bound = [0.95 * ones, 0.3 * ones, 0.15 * ones]

    lower_bound = [0.0]
    upper_bound = [0.95]

    for i_p in range(len(parameters)):
        parameter = parameters[i_p]
        # parameter_l = parameter-2*parameter/4
        # parameter_u = parameter+2*parameter/4
        parameter_l = parameter / 2
        parameter_u = parameter * 2
        lower_bound.append(parameter_l)
        upper_bound.append(parameter_u)
    # lower_bound = [lower_bound[i] * ones for i in range(len(lower_bound))]
    # upper_bound = [upper_bound[i] * ones for i in range(len(upper_bound))]

    # lower_bound = [0.0 * ones, 0.0 * ones, 0.065 * ones, 0.001 * ones, 0.1 * ones]
    # upper_bound = [0.95 * ones, 0.2 * ones, 0.8 * ones, 0.02 * ones, 0.7 * ones]

    # lower_bound = [0.0 * ones, 0.0 * ones, 0.065 * ones, 0.0065 * ones, 0.2 * ones]
    # upper_bound = [0.95 * ones, 0.2 * ones, 0.8 * ones, 0.0065 * ones, 0.2 * ones]

    # lower_bound = [0.0]
    # upper_bound = [0.95]

    lower_bounds = np.array([np.tile(lower_bound[i], number_control_change_times) for i in range(number_time_dependent_controls)]).flatten()
    for i in range(number_time_dependent_controls, len(lower_bound)):
        lower_bounds = np.append(lower_bounds, lower_bound[i])

    upper_bounds = np.array([np.tile(upper_bound[i], number_control_change_times) for i in range(number_time_dependent_controls)]).flatten()
    for i in range(number_time_dependent_controls, len(upper_bound)):
        upper_bounds = np.append(upper_bounds, upper_bound[i])

    # upper_bounds = np.append(upper_bounds, np.array(upper_bound[number_time_dependent_controls:]).flatten())

    # # bounds on initial conditions
    # y0_lower = 0.*y0
    # onesy0 = 1000. * ones
    # y0_upper = np.array([S, onesy0, onesy0, onesy0, onesy0, onesy0, onesy0, onesy0, onesy0, onesy0]).flatten("C")

    # y0_lower = y0
    # y0_upper = y0
    #
    # lower_bounds = np.append(lower_bounds, y0_lower)
    # upper_bounds = np.append(upper_bounds, y0_upper)

    bounds = Bounds(lower_bounds, upper_bounds, keep_feasible=True)

    # ######################## create the control objective and constraint, and solve the optimization problem
    objective_grad = grad(objective)
    objective_hess = hessian_vector_product(objective)

    # def hessp(flat_args, p, args):
    #     objective_hess(flat_args, args, p)

    constraint_grad = grad(constraint)
    constraint_hess = hessian_vector_product(constraint)
    nonlinear_constraint = NonlinearConstraint(constraint, -np.inf, 1000., jac=constraint_grad)

    controls_init = flat_args.copy()

    print("bounds length = ", len(upper_bounds), "controls_init", len(controls_init))

    # OptRes = minimize(fun=objective, x0=controls_init, args=(args,), method="l-bfgs-b", jac=objective_grad,
    #                      bounds=bounds, options={'maxiter':100, 'gtol':1e-2, 'disp': True})
    #
    OptRes = minimize(fun=objective, x0=controls_init, method="SLSQP", jac=objective_grad,
                      bounds=bounds,
                      options={'maxiter': 50, 'iprint': 10, 'disp': True})

    # cons_J = grad(constraint)
    # nonlinear_constraint = NonlinearConstraint(constraint, -np.inf, 1000., jac=cons_J, hess=BFGS())

    # OptRes = minimize(fun=objective, x0=controls_init, args=(args,), method="trust-constr", jac=objective_grad,
    #                   hessp=BFGS(), constraints=nonlinear_constraint, bounds=bounds,
    #                   options={'maxiter':100, 'gtol':1e-2, 'xtol':1e-2, 'initial_barrier_parameter':1e-1,
    #                            'disp': True, 'verbose':3})

    # OptRes = minimize(fun=objective, x0=controls_init, method="trust-constr",
    #                   jac=objective_grad, hessp=objective_hess, bounds=bounds,
    #                   options={'maxiter':20, 'gtol':1e-2, 'disp':True, 'verbose':3})

    print("optimization result = ", OptRes)

    print("state name = ", state_name)

    # compute the optimal solution
    controls_opt = unflatten(OptRes["x"])
    solution_opt = RK2(model.seir, y0, t_total, parameters, controls_opt)

    # compute the effective reproduction number
    Rt_opt = model.reproduction(t_total, parameters, controls_opt, solution_opt)

    # save data
    today = simulation_first_confirmed + len(data_confirmed)
    np.savez(savefilename, configurations=configurations, parameters=parameters,
             simulation_first_confirmed=simulation_first_confirmed, today=today,
             solution=solution, controls=controls, Rt=Rt,
             solution_opt=solution_opt, controls_opt=controls_opt, Rt_opt=Rt_opt)

    # print("# total infected = ", N_total - np.sum(solution_opt[ten_death_day+len(deaths), :number_group]),
    #       "# total death = ", np.sum(solution_opt[ten_death_day+len(deaths), 7 * number_group:8 * number_group]),
    #       "# hospitalized = ", np.sum(solution_opt[ten_death_day+len(deaths), 5 * number_group:6 * number_group]))
