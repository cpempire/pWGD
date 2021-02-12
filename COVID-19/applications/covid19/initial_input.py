
from __future__ import absolute_import, division, print_function

import autograd.numpy as np
from autograd.numpy import multiply as ewm

# the extended SEIR compartmental model with
# (1) vector for age (9 groups) and risk (2 groups) stratification, and
# (2) time dependent control variables

import os
if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isdir("figure"):
    os.mkdir("figure")

import sys
sys.path.append("../../")
from utils import *

import pickle

model_type = "scalar"
# model_type = "vector"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fips", nargs='?', const=48, default=48, type=int, help="specify fips of states")
args = parser.parse_args()

# Texas: 48, California: 6, New York: 36, New Jersey: 34
fips = args.fips

states = pickle.load(open("../../../data/states_dictionary_moving_average", 'rb'))

# population 0-4, 5-9, ..., 80-84, 85+
population = states[fips]['population']
N_total = np.sum(population)

# print("fips = ", fips, "population = ", N_total)

state_name = states[fips]['name']

# # # Texas
# death_by_age = np.array([0., 2., 7., 9., 19., 45, 87, 80, 163])
# death_by_age = death_by_age/np.sum(death_by_age)

# # New York May 21
# https://covid19tracker.health.ny.gov/views/NYS-COVID19-Tracker/NYSDOHCOVID-19Tracker-Fatalities?%3Aembed=yes&%3Atoolbar=no&%3Atabs=n
death_by_age = np.array([4., 10., 85., 307., 815., 2204., 4504, 6005, 5910+3123+9])
death_by_age = death_by_age/np.sum(death_by_age)

# age = np.linspace(0, 80, 9)+4.5
# coeff = np.polyfit(age, death_by_age, 2)
# death_by_age_regression = np.poly1d(coeff)(age)
#
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(age, death_by_age, 'x', label='data')
# plt.plot(age, death_by_age_regression, 'o', label='regression')
# plt.legend()
# plt.show()

if model_type is "scalar":

    filename_prex = "figure/initial_scalar_" + state_name + "_"
    savefilename = "data/initial_scalar_solution_" + state_name

    # ###### population, age and risk distributions specific to a given state/county/country
    # data for texas
    number_age_group = 1
    number_risk_group = 1
    number_group = number_age_group * number_risk_group

    # ##### parameters
    # important parameters to adjust that leads to reasonable basis reproduction number R0 and initial doubling time

    # transmission rate, fitted to give basic reproduction number (beta/gamma) and initial growth rate (beta-gamma)
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6962332/
    beta = 1.
    # # NY and NJ has more dense population, beta fitted according to the growth rate of confirmed
    # if fips == 34:
    #     beta = 1.5  # S to E and Q
    # elif fips == 36:
    #     beta = 1.7

    # latent rate, from infection to shedding (become infectious)
    # data from https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-30-COVID19-Report-13.pdf
    sigma = 1. / 3  # E to I

    # hospitalized rate, from shedding to hospitalization
    # https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf
    eta_I = 1. / 6  # I to H
    eta_Q = 1./(1./sigma + 1./eta_I)  # Q to H 1/(latent period/2 + shedding to hospitalization peiord)

    # ### deceased rate
    mu = 1. / 12  # H to D

    # data from https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf
    # data from https://theconversation.com/how-long-are-you-infectious-when-you-have-coronavirus-135295
    # most infectious period 1-3 days before symptom onset, and remains infectious 7 day after, so in average 2+5 = 7
    gamma_I = 1. / 3  # I to R
    gamma_A = 1. / 3  # A to R
    gamma_H = 1. / 14 # H to R
    # data from assumption 1 / (latent period + infectious period)
    gamma_Q = 1./(1./sigma + 1./gamma_I)  # Q to R 1/(5/2+9)

    # E to I proportion
    # estimate from Italy, 50%-75% are asymptomatic, https://www.bmj.com/content/368/bmj.m1165
    # data from https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf

    # constant for infected (confirmed) hospitalization ratio kappa / (kappa + tau(t-1/sigma))
    kappa = 0.04
    # hospitalized fatality ratio, H to D
    HFR = 0.2

    # Q to H proportion, assumption likely p_EI * p_IH
    # prop_QH = IFR/(100*HFR)

    # # relative transmission ratio, between the asymptomatic A and symptomatic I transmission rate to S
    # # data from China, https://science.sciencemag.org/content/sci/suppl/2020/03/13/science.abb3221.DC1/abb3221_Li_SM_rev.pdf
    # theta = 0.5

    # # data from EU https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-04-23-COVID19-Report-16.pdf
    # theta = 0.1   # a big range investigated from 0 - 1

    # ratio of pre-isolation infectious period, only in action if quarantine is used
    # data from EU, https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-04-23-COVID19-Report-16.pdf
    delta = 0.5

    # #### initial seeding and simulation time step
    # proportion of each age and risk group
    population_proportion = 1.

    # # susceptible population in each age and risk group
    S = np.array([ewm(N_total, population_proportion)])

    # exposed at the beginning
    E = np.array([100.])
    # S, E, Q, A, I, H, R, D, Tc, Tu # Tc/Tu is the total reported/unreported positive cases
    zeros = np.zeros(1)
    # Tc = np.ones(1) * data_confirmed[0]
    y0 = np.array([S, E, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros]).flatten("C")

    # # set data from solution to fitted death data
    # data = np.load("data/find_initial.npz")
    # solution_opt = data["solution_opt"]
    # today = data["today"]
    # y0 = solution_opt[today, :]

    # #### control variable for social distancing, isolution, and quarantine, in [0, 1], 0 means no control, 1 full control

    # control frequency and time
    number_days_per_control_change = 1
    number_control_change_times = 80
    number_control_days = number_days_per_control_change * number_control_change_times

    t_control = np.linspace(0, number_control_days, number_control_change_times)

    # time interval and steps
    number_days = number_control_days
    t_total = np.linspace(0, number_days, 1 * number_days + 1)

    # t_control = np.linspace(25, 25 + (number_control_change_times - 1) * number_days_per_control_change,
    #                         number_control_change_times)

    alpha = np.array([0.0 for i in range(number_control_change_times)])  # S to E social distancing
    q = 0.1
    tau = 0.5
    HFR = 0.2
    kappa = 0.2
    # q = np.array([0.0 for i in range(number_control_change_times)])  # S to Q quarantine
    # tau = np.array([0.1 for i in range(number_control_change_times)])  # E to I confirmation ratio, lower bound is 100 * IFR * HFR
    # HFR = np.array([0.2 for i in range(number_control_change_times)])
    # kappa = np.array([0.04 for i in range(number_control_change_times)])

    number_time_dependent_controls = 1

    # #################### collect the inputs for model construction
    parameters = (q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q)

    controls = (alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q)

    configurations = (y0, t_total, N_total, number_group, population_proportion,
                      t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls)

elif model_type is "vector":

    filename_prex = "figure/initial_vector_" + state_name + "_"
    savefilename = "data/initial_vector_solution_" + state_name

    # ###### population, age and risk distributions specific to a given state/county/country
    # data for texas
    number_age_group = 9
    number_risk_group = 2
    number_group = number_age_group * number_risk_group
    ones = np.ones(number_group)

    # total number of population
    # N = N_total * ones  # ewm(N_total, np.tile(age_distribution, 2))

    # expected lost quality life years (rescaled by the mean 40) given life expectancy = 79,
    # high risk people have a quality reduction of 50%
    # https://www.ssa.gov/oact/STATS/table4c6.html
    expected_life_years = np.array([75, 65, 55, 45, 36, 27, 19, 12, 5])
    expected_lost_life_years = np.append(expected_life_years, expected_life_years * 0.5)/40

    # age distribution for 0-9, 10-19, ..., 70-79, 80+
    age_distribution = np.array([np.sum(population[2*i:2*(i+1)]) for i in range(9)])/N_total

    # high risk distribution, percentage of high risk population in each age group
    # data obtained by fitting quadratic polynormial with data for Austin-Round Rock from
    # https://sites.cns.utexas.edu/sites/default/files/cid/files/austin_relaxing_social_distancing.pdf?m=1587681780
    high_risk_distribution = np.array([0.11, 0.13, 0.15, 0.19, 0.23, 0.29, 0.36, 0.44, 0.54])
    low_risk_distribution = 1 - high_risk_distribution

    # ##### parameters
    # important parameters to adjust that leads to reasonable basis reproduction number R0 and initial doubling time
    # beta_E, beta_Q, transmission rate rescaled by contact matrix
    # prop_EI, the proportion of exposed to symptomatic cases
    #

    # transmission rate, fitted to give basic reproduction number and initial growth rate
    beta = 0.1 * ones  # S to E and Q

    # latent rate, from infection to shedding (become infectious)
    # data from https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-30-COVID19-Report-13.pdf
    sigma = 1. / 4 * ones  # E to I and A

    # hospitalized rate, from shedding to hospitalization
    # https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf
    eta_I = 1. / 6 * ones  # I to H
    eta_Q = 1./(.5/sigma + 1./eta_I)  # Q to H 1/(latent period/2 + shedding to hospitalization period)

    # ### deceased rate
    # mu = 1. / 14 * ones  # H to D
    # data from California and Washington, Table S2, https://www.medrxiv.org/content/10.1101/2020.04.12.20062943v1.full.pdf
    # mu = np.tile(1. / np.array([8., 8., 9., 10.4, 11.5, 12.8, 13.9, 14.4, 14.0]), 2)

    # # data from France, Table S3, https://www.medrxiv.org/content/10.1101/2020.04.20.20072413v2.full.pdf
    mu = 1. / 12 * ones  # np.tile(1. / np.array([14., 14., 14., 14., 14., 14., 14., 10.3, 8.6]), 2)

    # ### recovery rate (1/infectious period, from shedding to clearance)
    # # data from California and Washington, Table S3, https://www.medrxiv.org/content/10.1101/2020.04.12.20062943v1.full.pdf
    # gamma_I = 1. / 9 * ones  # I to R
    # gamma_A = 1. / 9 * ones  # A to R

    # data from https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf
    # data from https://theconversation.com/how-long-are-you-infectious-when-you-have-coronavirus-135295
    # most infectious period 1-3 days before symptom onset, and remains infectious 7 day after, so in average 2+5 = 7
    gamma_I = 1. / 7 * ones  # I to R
    gamma_A = 1. / 7 * ones  # A to R

    # data from California and Washington, Table S2, https://www.medrxiv.org/content/10.1101/2020.04.12.20062943v1.full.pdf
    gamma_H = 1. / 14 * ones  # np.tile(1. / np.array([6.8, 6.8, 7.6, 8.8, 9.7, 10.8, 11.9, 12.3, 12.0]), 2)

    # # data gamma_H from France, not directly available, https://www.medrxiv.org/content/10.1101/2020.04.20.20072413v2.full.pdf

    # data from assumption 1 / (incubation period/2 + infectious period)
    gamma_Q = 1./(.5/sigma + 1./gamma_I)  # Q to R 1/(5/2+9)

    # # E to I proportion
    # # estimate from Italy, 50%-75% are asymptomatic, https://www.bmj.com/content/368/bmj.m1165
    # # data from https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf
    # prop_EI = 0.82 * ones
    #
    # # I to H proportion
    # # (all data are for infection hospitalization ratio, not symptomatic hospitalization ratio)
    # # data from California and Washington, Table S3, computed from Beta distribution
    # # https://www.medrxiv.org/content/10.1101/2020.04.12.20062943v1.full.pdf
    # prop_IH = np.array([0.5, 0.5, 1.2, 3.9, 4.8, 9.3, 13.4, 18.9, 20.4]) / 100
    #
    # # # data from France, Table S1, https://www.medrxiv.org/content/10.1101/2020.04.20.20072413v2.full.pdf
    # # prop_IH = np.array([0.1, 0.1, 0.5, 1.0, 1.5, 2.8, 6.1, 9.6, 21.7]) / 100
    #
    # # # data from China, https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
    # # prop_IH = np.array([0., 0.04,1.04, 3.43, 4.25, 8.16, 11.8, 16.6, 18.4]) / 100
    #

    # import matplotlib.pyplot as plt
    # plt.figure()

    # # # data from UK, adjusted from China above
    # # https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf
    # prop_IH = np.array([0.1, 0.3, 1.2, 3.2, 4.9, 10.2, 16.6, 24.3, 27.3]) / 100
    prop_IH = np.array([0.1, 0.3, 1.2, 3.2, 4.9, 10.2, 16.6, 24.3, 34.3]) / 100

    # plt.plot(prop_IH, 'b.-', label="IHR")
    # plt.plot(prop_IH/(np.max(prop_IH)+0.01), 'r.-', label="0.01")
    # plt.plot(prop_IH/(np.max(prop_IH)+0.1), 'c.-', label="0.1")
    # plt.plot(prop_IH/(np.max(prop_IH)+0.2), 'm.-', label="0.2")
    # plt.plot(prop_IH/(np.max(prop_IH)+0.5), 'g.-', label="0.5")
    # plt.plot(prop_IH/(np.max(prop_IH)+1.), 'k.-', label="1")
    # plt.legend()
    # plt.show()

    # 90% are high risk cases
    prop_IH_low = 0.1 * prop_IH/low_risk_distribution
    prop_IH_high = 0.9 * prop_IH/high_risk_distribution
    prop_IH = np.array([prop_IH_low, prop_IH_high]).flatten()

    # prop_IH = np.array([prop_IH/10, prop_IH]).flatten()
    # IFR = np.array([0.002, 0.006, 0.03, 0.08, 0.15, 0.60, 2.2, 5.1, 9.3]) / 100
    # # HFR = np.divide(IFR, prop_IH)
    # # print("HFR", HFR)
    #
    # # https://www.cdc.gov/mmwr/volumes/69/wr/mm6915e3.htm?s_cid=mm6915e3_w
    # prop_IH = np.append(prop_IH / 10, prop_IH)  # 10X riskier for high risk population, 90%+ with underlying condition

    # data from New Jersey
    # https://www.nj.gov/health/cd/documents/topics/NCOV/COVID_Confirmed_Case_Summary.pdf
    #

    # constant for infected (confirmed) hospitalization ratio kappa / (kappa + tau(t-1/sigma))
    kappa = prop_IH

    # H to D proportion
    # data from California and Washington, F&M averaged from Figure 3, 17.8% in average
    # https://www.medrxiv.org/content/10.1101/2020.04.12.20062943v1.full.pdf
    HFR = np.array([1., 1.6, 2.4, 4.1, 7.5, 11.6, 19.0, 30.9, 46.6]) / 100

    # plt.plot(HFR, 'rx', label="HFR")
    # plt.legend()
    # plt.show()

    # # data from France, Table S2, https://www.medrxiv.org/content/10.1101/2020.04.20.20072413v2.full.pdf
    # HFR = np.array([0.6, 0.6, 1.4, 2.1, 3.6, 7.0, 13.2, 23.2, 38.4]) / 100

    HFR = np.tile(HFR, 2)  # once in hospital the same risk for low and high risk population

    # # Q to H proportion, assumption likely p_EI * p_IH
    # prop_QH = ewm(prop_EI, prop_IH)

    # ### here is how HFR=HFR is computed
    # # proportion (I + A) to D low risk, X 10 for high risk
    # IFR = np.array([0.00091668, 0.0021789, 0.03388, 0.25197, 0.64402])
    # # proportion I to H low risk, X 10 for high risk
    # YHR = np.array([0.0279, 0.0215, 1.3215, 2.8563, 3.3873]) / 100
    # # proportion H to D low risk
    # HFR = np.divide(np.divide(IFR, prop_EI), YHR)
    # print("HFR = ", HFR)
    # ###

    # # relative transmission ratio, between the asymptomatic A and symptomatic I transmission rate to S
    # # data from China, https://science.sciencemag.org/content/sci/suppl/2020/03/13/science.abb3221.DC1/abb3221_Li_SM_rev.pdf
    # theta = 0.5 * ones

    # # data from EU https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-04-23-COVID19-Report-16.pdf
    # theta = 0.1 * ones  # a big range investigated from 0 - 1

    # ratio of pre-isolation infectious period, only in action if quarantine is used
    # data from EU, https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-04-23-COVID19-Report-16.pdf
    delta = 0.5 * ones

    # #### initial seeding and simulation time step
    # proportion of each age and risk group
    population_proportion = np.array([ewm(age_distribution, 1 - high_risk_distribution),
                     ewm(age_distribution, high_risk_distribution)]).flatten("C")

    # # susceptible population in each age and risk group
    S = ewm(N_total, population_proportion)

    # one exposed in age group 30-39
    # S, E, Q, A, I, H, R, D, Tc, Tu # Tc/Tu is the total reported/unreported positive cases
    zeros = np.zeros(number_group)
    E = 10 + zeros
    y0 = np.array([S, E, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros]).flatten("C")

    # #################### collect the inputs for model construction
    parameters = (HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q)

    # IHR = np.array([0.2 for i in range(number_control_change_times)])
    # HFR = np.array([0.2 for i in range(number_control_change_times)])

    number_time_dependent_controls = 3

    # #### control variable for social distancing, isolution, and quarantine, in [0, 1], 0 means no control, 1 full control

    # control frequency and time
    number_days_per_control_change = 7
    number_control_change_times = 10
    number_control_days = number_days_per_control_change * number_control_change_times

    t_control = np.linspace(0, number_control_days, number_control_change_times)

    # time interval and steps
    number_days = number_control_days + 30
    t_total = np.linspace(0, number_days, 1 * number_days + 1)

    # t_control = np.linspace(25, 25 + (number_control_change_times - 1) * number_days_per_control_change,
    #                         number_control_change_times)

    alpha = np.array([0.0 for i in range(number_control_change_times)])  # S to E social distancing
    q = np.array([0.0 for i in range(number_control_change_times)])  # S to Q quarantine
    tau = np.array([0.1 for i in range(number_control_change_times)])  # E to I confirmation ratio

    controls = (alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q)

    configurations = (y0, t_total, N_total, number_group, population_proportion,
                      t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls)
