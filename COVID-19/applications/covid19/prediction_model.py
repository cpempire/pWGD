
from __future__ import absolute_import, division, print_function

import time
import matplotlib.pyplot as plt

import sys
sys.path.append("../../")
from utils.interpolation import piecewiseConstant, piecewiseLinear, cubicSpline
from utils.integration import Euler, RK2, RK4

import autograd.numpy as np
from autograd.numpy import multiply as ewm


class Model:

    def __init__(self, configurations, parameters, controls):

        y0, t_total, N_total, number_group, population_proportion, \
        t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

        self.N_total = N_total
        self.number_group = number_group
        self.t_control = t_control
        self.number_time_dependent_controls = number_time_dependent_controls

        self.configurations = configurations
        self.parameters = parameters
        self.controls = controls

        self.interpolation = piecewiseLinear

        if number_group > 1:
            # contact matrix
            school_closure = True

            # calendar from February 15th
            weekday = [2, 3, 4, 5, 6]
            # calendar from April 1st
            # weekday = [0, 1, 2, 5, 6]
            # calendar from May 1st
            # weekday = [0, 3, 4, 5, 6]
            calendar = np.zeros(1000 + 1, dtype=int)
            # set work days as 1 and school days as 2
            for i in range(1001):
                if np.remainder(i, 7) in weekday:
                    calendar[i] = 1
                    if not school_closure:  # and i < 45
                        calendar[i] = 2
            self.calendar = calendar

            contact = np.load("../../utils/parameters/contact/contact_matrix.npz")
            self.c_home = contact["home"]
            self.c_school = contact["school"]
            self.c_work = contact["work"]
            self.c_other = contact["other"]

            self.contact_full = self.c_home + 5. / 7 * ((1 - school_closure) * self.c_school + self.c_work) + self.c_other

    def contact_rate(self, t):

        if self.number_group == 1:
            return 1.
        else:
            # ###### definition of the contact rate

            # contact_full = np.block([ewm(contact_full, np.tile(1-high_risk_distribution, [number_age_group, 1])),
            #                          ewm(contact_full, np.tile(high_risk_distribution, [number_age_group, 1]))])
            # contact_full = np.tile(contact_full, [number_risk_group, 1])

            # contact_full = np.tile(contact_full, [2, 2])

            # contact_full = 5*np.ones((number_group, number_group))

            if self.calendar[np.int(np.floor(t))] == 2:
                contact = self.c_home + self.c_school + self.c_work + self.c_other
            elif self.calendar[np.int(np.floor(t))] == 1:
                contact = self.c_home + self.c_work + self.c_other
            else:
                contact = self.c_home + self.c_other
            # else:
            #     contact = c_home + 0.1 * (c_work + c_other)
            # if calendar[np.int(np.floor(t))] == 1:
            #     contact = c_home + 0.1*(c_work + c_other)
            # else:
            #     contact = c_home

            # # construct contact matrix by splitting each age group into two by low and high risk proportion
            # contact = np.block([ewm(contact, np.tile(1 - high_risk_distribution, [number_age_group, 1])),
            #                          ewm(contact, np.tile(high_risk_distribution, [number_age_group, 1]))])
            # contact = np.tile(contact, [number_risk_group, 1])

            contact = np.tile(contact, [2, 2])

            # constant contact
            # contact = 10*np.ones((number_group, number_group))

            return contact

    def proportion2factor(self, proportion, r1, r2):
        # define a function to transform proportion to factor, e.g., from prop_EI to tau with r1=sigma_I, r2 = sigma_A
        factor = np.divide(ewm(r2, proportion), r1 + ewm(r2 - r1, proportion))

        return factor

    def seir(self, y, t, parameters, controls, stochastic=False):
        # define the right hand side of the ode systems given state y, time t, parameters, and controls

        if self.number_group > 1:
            y = y.reshape((10, self.number_group))

        S, E, Q, A, I, H, R, D, Tc, Tu = y

        q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = parameters
        # alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = controls
        # alpha, q, tau, HFR, kappa, beta, _, _, _, _, _, _, _, _, _ = controls

        alpha = self.interpolation(t, self.t_control, controls)

        # tau_p = self.interpolation(t - np.max(1./sigma), self.t_control, controls[2])
        # tau_p = self.interpolation(t, self.t_control, controls[2])

        # IHR = np.divide(kappa, np.max(kappa) + tau_p)

        IHR = kappa

        QHR = ewm(tau, IHR)

        # gamma_A = gamma_I

        pi = self.proportion2factor(IHR, eta_I, gamma_I)
        nu = self.proportion2factor(HFR, mu, gamma_H)
        rho = self.proportion2factor(QHR, eta_Q, gamma_Q)

        contact = self.contact_rate(t)

        # theta_I = 2 - tau
        # theta_A = 1 - tau

        theta_I = 1. - 0*tau
        theta_A = 1. - 0*tau
        delta = 1. + 0*delta

        # print("q = ", q, "delta = ", delta, "N_total = ", self.N_total, "contact = ", contact)
        C_E = ewm(1-alpha,
                  ewm(1-q, ewm(delta, np.dot(contact, ewm(theta_I, np.divide(I, self.N_total))))) +
                  np.dot(contact, ewm(theta_A, np.divide(A, self.N_total)))
                  )

        C_Q = ewm(1-alpha,
                  ewm(q, ewm(delta, np.dot(contact, ewm(theta_I, np.divide(I, self.N_total)))))
                  )

        if stochastic:
            zeros = np.zeros(np.size(S))
            S = np.max([zeros, S], axis=0)
            E = np.max([zeros, E], axis=0)
            Q = np.max([zeros, Q], axis=0)
            A = np.max([zeros, A], axis=0)
            I = np.max([zeros, I], axis=0)
            H = np.max([zeros, H], axis=0)

        P1 = ewm(beta, ewm(C_E, S))
        # print("P1 = ", P1, "S = ", S, "beta = ", beta, "C_E = ", C_E)
        P2 = ewm(beta, ewm(C_Q, S))
        P3 = ewm(tau, ewm(sigma, E))
        P4 = ewm(1 - tau, ewm(sigma, E))
        # print("tau, sigma, E, P4 = ", tau, sigma, E, P4)
        P5 = ewm(rho, ewm(eta_Q, Q))
        P6 = ewm(1 - rho, ewm(gamma_Q, Q))
        P7 = ewm(gamma_A, A)
        P8 = ewm(pi, ewm(eta_I, I))
        P9 = ewm(1 - pi, ewm(gamma_I, I))
        P10 = ewm(nu, ewm(mu, H))
        P11 = ewm(1 - nu, ewm(gamma_H, H))

        if stochastic:
            P1 = np.random.poisson(P1)
            P2 = np.random.poisson(P2)
            P3 = np.random.poisson(P3)
            P4 = np.random.poisson(P4)
            P5 = np.random.poisson(P5)
            P6 = np.random.poisson(P6)
            P7 = np.random.poisson(P7)
            P8 = np.random.poisson(P8)
            P9 = np.random.poisson(P9)
            P10 = np.random.poisson(P10)
            P11 = np.random.poisson(P11)

        dS = - P1 - P2
        dE = P1 - P3 - P4
        dQ = P2 - P5 - P6
        dA = P4 - P7
        dI = P3 - P8 - P9
        dH = P8 + P5 - P10 - P11
        dR = P7 + P9 + P11 + P6
        dD = P10
        dTc = P3 + P2   # + quarantined, P2
        dTu = P4
        dydt = np.array([dS, dE, dQ, dA, dI, dH, dR, dD, dTc, dTu]).flatten("C")

        return dydt

    def grouping(self, solution):
        # grouping different groups in solution into one group

        if self.number_group > 1:
            # solution_help = np.zeros((solution.shape[0], 10))
            # for i in range(10):
            #     solution_help[:, i] = np.sum(solution[:, i * self.number_group:(i+1) * self.number_group], axis=1)
            # solution = solution_help

            solution_help = []
            for i in range(10):
                solution_help = np.append(solution_help, np.sum(solution[:, i * self.number_group:(i+1) * self.number_group], axis=1))
            solution = np.reshape(solution_help, (10, solution.shape[0])).T

        return solution

    def reproduction(self, t, parameters, controls, solution):
        # effective reproduction number at time t
        # https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2009.0386
        # The construction of next-generation matrices for compartmental epidemic models

        t_set = t
        number_time = len(t)
        if number_time == 1:
            t_set = [t]

        Rt = np.zeros(number_time)
        for n_t in range(number_time):
            t = t_set[n_t]

            q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = parameters
            # alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = controls
            # alpha, q, tau, HFR, kappa, beta, _, _, _, _, _, _, _, _, _ = controls

            alpha = self.interpolation(t, self.t_control, controls)

            # tau_p = self.interpolation(t - np.max(1./sigma), self.t_control, controls[2])
            # tau_p = self.interpolation(t, self.t_control, controls[2])

            # IHR = np.divide(kappa, np.max(kappa) + tau_p)

            IHR = kappa

            QHR = ewm(tau, IHR)

            pi = self.proportion2factor(IHR, eta_I, gamma_I)

            # transform the parameter in the vector format
            ones = np.ones(self.number_group)

            alpha, q, tau, pi, beta, delta, kappa, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = \
            ewm(alpha,ones), ewm(q,ones), ewm(tau,ones), ewm(pi,ones), ewm(beta,ones), ewm(delta,ones), ewm(kappa,ones), \
            ewm(sigma,ones), ewm(eta_I,ones), ewm(eta_Q,ones), ewm(mu,ones), ewm(gamma_I,ones), ewm(gamma_A,ones), ewm(gamma_H,ones), ewm(gamma_Q,ones)

            # theta_I = 2 - tau
            # theta_A = 1 - tau

            theta_I = 1. - 0 * tau
            theta_A = 1. - 0 * tau
            delta = 1. + 0 * delta

            S = solution[n_t, :self.number_group]

            zeros = np.zeros((self.number_group, self.number_group))
            ItoS = np.diag(beta) * np.diag(1.-alpha) * np.diag(1.-q) * np.diag(delta) * np.dot(self.contact_rate(t), np.diag(theta_I)) * np.diag(np.divide(S, self.N_total))
            AtoS = np.diag(beta) * np.diag(1.-alpha) * np.dot(self.contact_rate(t), np.diag(theta_A)) * np.diag(np.divide(S, self.N_total))
            T = np.block([
                [zeros, AtoS, ItoS],
                [zeros, zeros, zeros],
                [zeros, zeros, zeros]
            ])

            Sigma = np.block([
                [-np.diag(sigma), zeros, zeros],
                [np.diag(1.-tau)*np.diag(sigma), -np.diag(gamma_A), zeros],
                [np.diag(tau)*np.diag(sigma), zeros, -np.diag(pi)*np.diag(eta_I)-np.diag(1.-pi)*np.diag(gamma_I)]
            ])

            w, _ = np.linalg.eig(-np.dot(T, np.linalg.inv(Sigma)))

            Rt[n_t] = np.max(w)

        return Rt

    # define plot the solution
    def plotsolution(self, t, solution, solution_opt=None, filename_prex=None):
        # plot solutions

        solution = self.grouping(solution)

        plt.figure(1)
        plt.semilogy(t, solution[:, 0], 'b', label='$S$: susceptible')
        plt.semilogy(t, solution[:, 1], 'g', label='$E$: exposed')
        plt.semilogy(t, solution[:, 2], 'r', label='$Q$: quarantined')
        plt.semilogy(t, solution[:, 3], 'k', label='$A$: unconfirmed')
        plt.semilogy(t, solution[:, 4], 'm', label='$I$: confirmed')
        plt.semilogy(t, solution[:, 5], 'y', label='$H$: hospitalized')
        plt.semilogy(t, solution[:, 6], 'g.-', label='$R$: recovered')
        plt.semilogy(t, solution[:, 7], 'c', label='$D$: deceased')
        plt.semilogy(t, solution[:, 8], 'm.-', label='$T_c$: total confirmed')
        plt.semilogy(t, solution[:, 9], 'k.-', label='$T_u$: total unconfirmed')
        plt.legend(loc='best')
        plt.xlabel('time t (days)')
        plt.ylabel('# cases')
        plt.grid()
        if filename_prex is not None:
            filename = filename_prex + "all_compartments.pdf"
            plt.savefig(filename)

        plt.figure(2)
        plt.semilogy(t, solution[:, 5], 'y', label='$H$: hospitalized')
        plt.semilogy(t, solution[:, 7], 'c', label='$D$: deceased')
        # plt.semilogy(t, solution[:, 8], 'm.-', label='$T_c$: total confirmed')
        # plt.semilogy(t, solution[:, 9], 'k.-', label='$T_u$: total unconfirmed')
        plt.legend(loc='best')
        plt.xlabel('time t (days)')
        plt.ylabel('# cases')
        plt.grid()
        if filename_prex is not None:
            filename = filename_prex + "hospitalized_deceased.pdf"
            plt.savefig(filename)

        if solution_opt is not None:

            solution_opt = self.grouping(solution_opt)

            plt.figure(1)
            plt.semilogy(t, solution_opt[:, 0], 'b--', label='$S$: susceptible')
            plt.semilogy(t, solution_opt[:, 1], 'g--', label='$E$: exposed')
            plt.semilogy(t, solution_opt[:, 2], 'r--', label='$Q$: quarantined')
            plt.semilogy(t, solution_opt[:, 3], 'k--', label='$A$: unconfirmed')
            plt.semilogy(t, solution_opt[:, 4], 'm--', label='$I$: confirmed')
            plt.semilogy(t, solution_opt[:, 5], 'y--', label='$H$: hospitalized')
            plt.semilogy(t, solution_opt[:, 6], 'g.--', label='$R$: recovered')
            plt.semilogy(t, solution_opt[:, 7], 'c--', label='$D$: deceased')
            plt.semilogy(t, solution_opt[:, 8], 'm:', label='$T_c$: total confirmed')
            plt.semilogy(t, solution_opt[:, 9], 'k:', label='$T_u$: total unconfirmed')

            plt.legend(loc='best')
            plt.xlabel('time t (days)')
            plt.ylabel('# cases')
            plt.grid()
            if filename_prex is not None:
                filename = filename_prex + "all_compartments.pdf"
                plt.savefig(filename)

            plt.figure(2)
            plt.semilogy(t, solution_opt[:, 5], 'y--', label='$H$: hospitalized')
            plt.semilogy(t, solution_opt[:, 7], 'c--', label='$D$: deceased')
            # plt.semilogy(t, solution_opt[:, 8], 'm:', label='$T_c$: total confirmed')
            # plt.semilogy(t, solution_opt[:, 9], 'k:', label='$T_u$: total unconfirmed')
            plt.legend(loc='best')
            plt.xlabel('time t (days)')
            plt.ylabel('# cases')
            plt.grid()
            if filename_prex is not None:
                filename = filename_prex + "hospitalized_deceased.pdf"
                plt.savefig(filename)

        # plt.show()
        # plt.close()


if __name__ == "__main__":

    from initial_input import configurations, parameters, controls

    model = Model(configurations, parameters, controls)

    # print("reproduction number = ", model.reproduction(parameters, controls))

    ######### solve the ode systems ########

    # t0 = time.time()
    # solution = odeint(seir, y0, t, args=(parameters,))
    # sol_odeint = solution.copy()
    # print("solve time by odeint ", time.time() - t0)

    # t0 = time.time()
    # solution = Euler(seir, y0, t, parameters, controls)
    # # np.savez(savefilename, t=t, solution=solution, controls=controls)
    # print("solve time by Euler method", time.time() - t0)
    # # # print("euler diff = ", np.linalg.norm(sol_odeint-solution)/np.linalg.norm(sol_odeint))

    y0, t_total, N_total, number_group, population_proportion, \
    t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

    t0 = time.time()
    solution = RK2(model.seir, y0, t_total, parameters, controls)
    print("solve time by RK2 method ", time.time() - t0)
    # print("RK2 diff = ", np.linalg.norm(sol_odeint-solution)/np.linalg.norm(sol_odeint))

    print("solution = ", solution[:, 5])

    model.plotsolution(t_total, solution)

    Rt = model.reproduction(t_total, parameters, controls, solution)
    plt.figure()
    plt.plot(Rt, '.-')
    plt.show()

    # t0 = time.time()
    # solution = RK4(seir, y0, t, parameters, controls)
    # print("solve time by RK4 method ", time.time() - t0)
    # print("RK4 diff = ", np.linalg.norm(sol_odeint-solution)/np.linalg.norm(sol_odeint))

    solution = model.grouping(solution)

    print("# total infected = ", model.N_total - solution[-1, 0],
          "# total death = ", solution[-1, 7],
          "# total positive = ", solution[-1, 8],
          "maximum # hospitalized = ", np.max(solution[:, 5]))

    # print("IFR separate = ", np.divide(solution[-1, 7*number_group:8*number_group],
    #                           S - solution[-1, :number_group]))
    print("IFR total = ", solution[-1, 7]/
                                 (model.N_total - solution[-1, 0]))

    # compute growth rate at properly chosen time intervals
    t0, t1 = 10, 20
    E0, E1 = solution[t0, 1], solution[t1, 1]
    print("exposed growth rate beta for exp(beta*t) = ", np.log(E1/E0)/(t1-t0), "doubling time = ", np.log(2)/(np.log(E1/E0)/(t1-t0)))
    t0, t1 = 20, 30
    I0, I1 = solution[t0, 4], solution[t1, 4]
    print("symptomatic growth rate beta for exp(beta*t) = ", np.log(I1/I0)/(t1-t0), "doubling time = ", np.log(2)/(np.log(I1/I0)/(t1-t0)))
    t0, t1 = 30, 40
    D0, D1 = solution[t0, 7], solution[t1, 7]
    print("deceased growth rate beta for exp(beta*t) = ", np.log(D1/D0)/(t1-t0), "doubling time = ", np.log(2)/(np.log(D1/D0)/(t1-t0)))

    # for k in range(10):
    #     t0 = time.time()
    #     solution = RK2(seir_stochastic, y0, t, parameters, controls)
    #     print("solve time by RK2 method", time.time() - t0)
    #
    #     print("# total infected = ", N_total - np.sum(solution[-1, :number_group]),
    #           "# total death = ", np.sum(solution[-1, 7*number_group:8*number_group]),
    #           "maximum # hospitalized = ", np.max(np.sum(solution[:, 5*number_group:6*number_group], axis=1)))
    #
    #     plotsolution(t, solution)

    # plt.show()

    # # find initial
    # data = np.load("data/find_initial.npz")
    # today = data["today"]
    # ten_death_day = data["ten_death_day"]
    #
    # plt.figure()
    # xaxis = np.array(range(today, solution.shape[0]+today))
    # plt.semilogy(xaxis, solution[:, 7], 'b.-')
    #
    # import sys
    # sys.path.append("../../../")
    # from data import *
    #
    # states = nyt_states_data(start_date='2020-01-21', end_date='2020-04-28')
    # fips = 48
    # first = np.where(states[fips]["deaths"] > 10)[0][0]
    # xaxis = np.array(range(ten_death_day, today))
    # print("deaths = ", states[fips]["deaths"][first:])
    # plt.semilogy(xaxis, states[fips]["deaths"][first:], 'r.-')
    #
    # print("days from 10 deaths = ", len(states[fips]["deaths"][first:]))
    #
    # plt.show()