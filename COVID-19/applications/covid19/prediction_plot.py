
from __future__ import absolute_import, division, print_function

import autograd.numpy as np
from autograd.numpy import multiply as ewm
import scipy.stats

import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from datetime import datetime, timedelta

import sys
sys.path.append("../../")
from utils import *

import pickle
# from initial_input import *

from prediction_model import Model

model_type = "scalar"
# model_type = "vector"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fips", nargs='?', const=48, default=48, type=int, help="specify fips of states")
args = parser.parse_args()

# Texas: 48, California: 6, New York: 36, New Jersey: 34
fips = args.fips

states = pickle.load(open("../../../data/states_dictionary", 'rb'))
states_moving_average = pickle.load(open("../../../data/states_dictionary_moving_average", 'rb'))
state_name = states[fips]["name"]

if model_type is "scalar":
    filename_prex = "figure/prediction_scalar_" + state_name + "_"
else:
    filename_prex = "figure/prediction_vector_" + state_name + "_"

# ############# plot death data
first_confirmed = np.where(states[fips]["positive"] > 100)[0][0]
data_confirmed = states[fips]["positive"][first_confirmed:]
data_confirmed_moving_average = states_moving_average[fips]["positive"][first_confirmed:]

first_hospitalized = np.where(states[fips]["hospitalizedCurrently"] > 10)[0][0]
lag_hospitalized = first_hospitalized - first_confirmed
data_hospitalized = states[fips]["hospitalizedCurrently"][first_hospitalized:]
data_hospitalized_moving_average = states_moving_average[fips]["hospitalizedCurrently"][first_hospitalized:]

first_deceased = np.where(states[fips]["death"] > 10)[0][0]
lag_deceased = first_deceased - first_confirmed
data_deceased = states[fips]["death"][first_deceased:]
data_deceased_moving_average = states_moving_average[fips]["death"][first_deceased:]

# population 0-4, 5-9, ..., 80-84, 85+
N_total = np.sum(states[fips]['population'])

# print("fips = ", fips, "population = ", N_total)

state_name = states[fips]['name']

data = np.load("data/initial_scalar_solution_" + state_name +".npz", allow_pickle=True)
solution_opt = data["solution_opt"]
controls_opt = data["controls_opt"]
configurations = data["configurations"]
today = data["today"]
simulation_first_confirmed = data["simulation_first_confirmed"]

y0, t_total, N_total, number_group, population_proportion, \
t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

alpha_opt, q_opt, tau_opt, HFR_opt, kappa_opt, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = controls_opt
y0 = solution_opt[today, :]

# #################### collect the inputs for model construction
parameters = np.array([q_opt, tau_opt, HFR_opt, kappa_opt, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q])

# IHR = np.array([0.2 for i in range(number_control_change_times)])
# HFR = np.array([0.2 for i in range(number_control_change_times)])

number_time_dependent_controls = 1

# #### control variable for social distancing, isolution, and quarantine, in [0, 1], 0 means no control, 1 full control

# control frequency and time
number_days_per_control_change = 1
number_control_change_times = 28
number_control_days = number_days_per_control_change * number_control_change_times

t_control = np.linspace(0, number_control_days, number_control_change_times)

# time interval and steps
number_days = number_control_days
t_total = np.linspace(0, number_days, 1 * number_days + 1)

# t_control = np.linspace(25, 25 + (number_control_change_times - 1) * number_days_per_control_change,
#                         number_control_change_times)

alpha = np.array([alpha_opt[-1] for i in range(number_control_change_times)])  # S to E social distancing
# q = np.array([q_opt[-1] for i in range(number_control_change_times)])  # S to Q quarantine
# tau = np.array([tau_opt[-1] for i in
#                 range(number_control_change_times)])  # E to I confirmation ratio, lower bound is 100 * IFR * HFR
# HFR = np.array([HFR_opt[-1] for i in range(number_control_change_times)])  # S to Q quarantine
# kappa = np.array([kappa_opt[-1] for i in range(number_control_change_times)])  # S to Q quarantine

controls = alpha

configurations = (y0, t_total, N_total, number_group, population_proportion,
                  t_control, number_days_per_control_change, number_control_change_times,
                  number_time_dependent_controls)

model = Model(configurations, parameters, controls)

# Rt = model.reproduction(t_total, parameters, controls, solution)

mean = parameters
std = 0.1 * mean

number_sample = 100
sampling = lognormal(mean, std)
samples = sampling.sample(number_sample)

print("samples = ", samples[0])

solutions = []
for i in range(number_sample):
    solution = RK2(model.seir, y0, t_total, samples[i, :], controls, stochastic=True)
    solution = model.grouping(solution)
    solutions.append(solution)

    print("simulation # ", i, " / ", str(number_sample))

solution_opt = model.grouping(solution_opt)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


# # plot data, moving average, optimized simulation, and prediction
time_delta = timedelta(days=1)
stop_date = datetime(2020, 5, 20)

datas = [data_deceased, data_hospitalized, data_confirmed]
datas_moving_average = [data_deceased_moving_average, data_hospitalized_moving_average, data_confirmed_moving_average]
indices = [7, 5, 8]
labels = ["simulation", "data", "data moving average", "mean of prediction"]
daily_titles = ["daily-deceased", "current-hospitalized", "daily-confirmed"]
total_titles = ["total-deceased", "current-hospitalized", "total-confirmed"]
for ind in range(3):
    data = datas[ind]
    data_moving_average = datas_moving_average[ind]
    index = indices[ind]

    start_date = stop_date - timedelta(len(data)-1)
    dates = mdates.drange(start_date, stop_date, time_delta)

    future_date = stop_date + timedelta(len(t_total)-1)
    dates_prediction = mdates.drange(stop_date, future_date, time_delta)

    # plot daily
    fix, ax = plt.subplots()
    ax.plot(dates, np.diff(solution_opt[today-len(data)+1:today+1, index]), 'b.-', label=labels[0])
    ax.plot(dates, np.diff(data), 'r.-', label=labels[1])
    ax.plot(dates, np.diff(data_moving_average), 'k.-', label=labels[2])

    daily = []
    for i in range(number_sample):
        solution = solutions[i]
        ax.plot(dates_prediction, np.diff(solution[:, index]), '--', linewidth=0.5)
        daily.append(np.diff(solution[:, index]))
    daily = np.array(daily)
    number_days = len(daily[0,:])
    daily_average = np.zeros(number_days)
    daily_plus = np.zeros(number_days)
    daily_minus = np.zeros(number_days)
    for i in range(number_days):
        daily_sort = np.sort(daily[:,i])
        # daily_average[i], daily_plus[i], daily_minus[i] = mean_confidence_interval(daily[:,i])
        daily_average[i], daily_plus[i], daily_minus[i] = \
            np.mean(daily_sort), daily_sort[np.int(2.5/100*number_sample)], daily_sort[np.int(97.5/100*number_sample)]

    ax.plot(dates_prediction, daily_average, '.-', linewidth=2, label=labels[3])
    ax.fill_between(dates_prediction, daily_minus, daily_plus, color='gray', alpha=.2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.legend(loc='best')
    ax.grid(True)
    plt.title(daily_titles[ind])
    plt.legend()
    filename = filename_prex + daily_titles[ind] + ".pdf"
    plt.savefig(filename)

    # # plot total
    fix, ax = plt.subplots()
    ax.plot(dates, solution_opt[today-len(data)+2:today+1, index], 'b.-', label=labels[0])
    ax.plot(dates, data[1:], 'r.-', label=labels[1])
    ax.plot(dates, data_moving_average[1:], 'k.-', label=labels[2])

    total = []
    for i in range(number_sample):
        solution = solutions[i]
        ax.plot(dates_prediction, solution[1:, index], '--', linewidth=0.5)
        total.append(solution[1:, index])
    total = np.array(total)
    number_days = len(total[0,:])
    total_average = np.zeros(number_days)
    total_plus = np.zeros(number_days)
    total_minus = np.zeros(number_days)
    for i in range(number_days):
        total_sort = np.sort(total[:,i])
        # total_average[i], total_plus[i], total_minus[i] = mean_confidence_interval(total[:,i])
        total_average[i], total_plus[i], total_minus[i] = \
            np.mean(total_sort), total_sort[np.int(2.5/100*number_sample)], total_sort[np.int(97.5/100*number_sample)]

    ax.plot(dates_prediction, total_average, '.-', linewidth=2, label=labels[3])
    ax.fill_between(dates_prediction, total_minus, total_plus, color='gray', alpha=.2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.legend(loc='best')
    ax.grid(True)

    plt.legend()
    plt.title(total_titles[ind])
    filename = filename_prex + total_titles[ind] + ".pdf"
    plt.savefig(filename)

    # plt.show()