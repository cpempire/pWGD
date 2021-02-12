import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
# # plot data, moving average, optimized simulation, and prediction

data = pickle.load(open("data/data_nSamples_128_isProjection_True_WGD.p", 'rb'))
d_average = data["d_average"]
plt.figure()
step = 20
for i in range(np.floor_divide(len(d_average), step)):
    label = "$\ell = $"+str(i*step)
    plt.plot(np.log10(np.sort(d_average[i*step])[::-1]), '.-', label=label)
plt.xlabel("r", fontsize=16)
plt.ylabel(r"$\log_{10}(|\lambda_r|)$", fontsize=16)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.savefig("figure/covid19_eigenvalues.pdf")
# plt.show()
plt.close()

from model import *

time_delta = timedelta(days=1)
stop_date = datetime(2020, 6, 6)
start_date = stop_date - timedelta(len(misfit.t_total))
dates = mdates.drange(start_date, stop_date, time_delta)


def plot_total(t, samples, ax):

    total = samples
    number_sample = len(total)
    dim = len(total[0, :])
    total_average = np.zeros(dim)
    total_plus = np.zeros(dim)
    total_minus = np.zeros(dim)
    for id in range(dim):
        total_sort = np.sort(total[:, id])
        # total_average[i], total_plus[i], total_minus[i] = mean_confidence_interval(total[:,i])
        id_1 = np.int(5 / 100 * number_sample)
        id_2 = np.int(95 / 100 * number_sample)
        total_average[id], total_plus[id], total_minus[id] = np.mean(total_sort[id_1:id_2]), total_sort[id_1], total_sort[id_2]

    ax.plot(t, total_average, '.-', linewidth=2, label="mean")
    ax.fill_between(t, total_minus, total_plus, color='gray', alpha=.2)


for i in range(20,21):

    filename = "data/particle_isProjection_False_"+"iteration_"+str(i*10)+".npz"
    data = np.load(filename)
    particles_WGD = data["particle"]

    filename = "data/particle_isProjection_True_"+"iteration_"+str(i*10)+".npz"
    data = np.load(filename)
    particles_pWGD = data["particle"]

    shape = particles_pWGD.shape
    print("i = ", i, "particles shape = ", shape)
    samples_pWGD = []
    solutions_pWGD = []
    samples_WGD = []
    solutions_WGD = []
    for j in range(shape[0]):
        for k in range(shape[1]):
            sample = particles_WGD[j, k, :]
            solution = misfit.solution(sample)
            samples_WGD.append((np.tanh(sample)+1)/2)
            solutions_WGD.append(solution)

            sample = particles_pWGD[j, k, :]
            solution = misfit.solution(sample)
            samples_pWGD.append((np.tanh(sample)+1)/2)
            solutions_pWGD.append(solution)

    solutions_WGD = np.array(solutions_WGD)
    solutions_pWGD = np.array(solutions_pWGD)
    samples_WGD = np.array(samples_WGD)
    samples_pWGD = np.array(samples_pWGD)

    # # # plot samples and solutions at one figure
    # ax1 = plt.subplot(2,2,1)
    # plt.title("WGD sample")
    # plot_total(dates, samples_WGD, ax1)
    # ax1.legend()
    #
    # ax2 = plt.subplot(2,2,2)
    # plt.title("WGD solution")
    # plot_total(dates, solutions_WGD, ax2)
    # ax2.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=4, label="data")
    # ax2.legend()
    #
    # ax3 = plt.subplot(2,2,3)
    # plt.xlabel("pWGD sample")
    # plot_total(dates, samples_pWGD, ax3)
    # ax3.legend()
    #
    # ax4 = plt.subplot(2,2,4)
    # plt.xlabel("pWGD solution")
    # plot_total(dates, solutions_pWGD, ax4)
    # ax4.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=4, label="data")
    # ax4.legend()
    #
    # filename = "figure/pWGDvsWGD_" + str(i*10) + ".pdf"
    # plt.savefig(filename)
    #
    # plt.close("all")

    # # plot samples
    ax1 = plt.subplot(2, 1, 1)
    plot_total(dates[:-1], samples_WGD[:, :-1], ax1)
    # ax1.plot(dates[:-1], opt_data["controls_opt"][0][:-1], 'k.-', label=r"$\hat{\alpha}$")
    plt.title("WGD social distancing", fontsize=16)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2)
    plot_total(dates[:-1], samples_pWGD[:,:-1], ax2)
    # ax2.plot(dates[:-1], opt_data["controls_opt"][0][:-1], 'k.-', label=r"$\hat{\alpha}$")
    plt.xlabel("pWGD social distancing", fontsize=16)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.grid(True)

    filename = "figure/samples_pWGDvsWGD_" + str(i*10) + ".pdf"
    plt.savefig(filename)

    plt.close("all")

    # # plot solutions
    ax1 = plt.subplot(2, 1, 1)
    plot_total(dates[:-1], solutions_WGD[:,:-1], ax1)
    ax1.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=2, label="data")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.grid(True)
    plt.title("WGD # hospitalized", fontsize=16)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    plot_total(dates[:-1], solutions_pWGD[:,:-1], ax2)
    ax2.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=2, label="data")

    plt.xlabel("pWGD # hospitalized", fontsize=16)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.grid(True)

    filename = "figure/solutions_pWGDvsWGD_" + str(i * 10) + ".pdf"
    plt.savefig(filename)
