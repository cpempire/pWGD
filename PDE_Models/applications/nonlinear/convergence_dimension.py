import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dSet = np.array([8, 16, 32, 64])
dPara = [81, 289, 1089, 4225]
NSet = [256, 256, 256, 256]
# NSet = [64, 64, 64, 64]
# NSet = [16, 16, 16, 16]

Ntrial = 10
Niter = 100

makerTrue = ['g.--', 'k.--', 'm.--', 'c.--']
makerFalse = ['b.--', 'r.--', 'y.--', 'b.--']
makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']
makerRMSEFalse = ['bs-', 'r*-', 'y<-', 'b>-']


######################################## SVGD ##################################

meanErrorL2normTrue = np.zeros((4, Ntrial, Niter))
varianceErrorL2normTrue = np.zeros((4, Ntrial, Niter))
meanErrorL2normFalse = np.zeros((4, Ntrial, Niter))
varianceErrorL2normFalse = np.zeros((4, Ntrial, Niter))
# meanErrorL2normSVGD = np.zeros((4, Ntrial, Niter))
# varianceErrorL2normSVGD = np.zeros((4, Ntrial, Niter))

labelTrue = ['pSVGD d=81', 'pSVGD d=289', 'pSVGD d=1089', 'pSVGD d=4225']
labelFalse = ['SVGD d=81', 'SVGD d=289', 'SVGD d=1089', 'SVGD d=4225']
fig1 = plt.figure(1)
fig2 = plt.figure(2)
for case in [0, 1, 2, 3]:

    d, N = dSet[case], NSet[case]

    for i in range(Ntrial):
        filename = "accuracy-" + str(d) + 'x' + str(d) + "/data/data_nDimensions_"+str((d+1)**2)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(True) + "_SVGD_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
        varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
        meanErrorL2normTrue[case, i, :] = meanErrorL2normTrue_i
        varianceErrorL2normTrue[case, i, :] = varianceErrorL2normTrue_i

        filename = "accuracy-" + str(d) + 'x' + str(d) + "/data/data_nDimensions_"+str((d+1)**2)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(False) + "_SVGD_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
        varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
        meanErrorL2normFalse[case, i, :] = meanErrorL2normFalse_i
        varianceErrorL2normFalse[case, i, :] = varianceErrorL2normFalse_i

    plt.figure(1)
    for i in range(Ntrial):
        plt.plot(np.log10(meanErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
        plt.plot(np.log10(meanErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
    plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normTrue[case,:,:]**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])

    plt.figure(2)
    for i in range(Ntrial):
        plt.plot(np.log10(varianceErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
        plt.plot(np.log10(varianceErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normFalse[case,:,:]**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
    plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normTrue[case,:,:]**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])

plt.figure(1)
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of mean)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_mean_SVGD.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean_SVGD.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()

plt.figure(2)
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of variance)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_variance_SVGD.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance_SVGD.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()


meanErrorDimensionTrue = meanErrorL2normTrue[:, :, -1]
meanErrorDimensionFalse = meanErrorL2normFalse[:, :, -1]
varianceErrorDimensionTrue = varianceErrorL2normTrue[:, :, -1]
varianceErrorDimensionFalse = varianceErrorL2normFalse[:, :, -1]

case = 0

fig1 = plt.figure(1)
for i in range(Ntrial):
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionFalse[:, i]), makerFalse[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionTrue[:, i]), makerTrue[case], alpha=0.2)

plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='SVGD')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pSVGD')
plt.figure(1)
plt.legend(fontsize=16)
plt.xlabel("$\log_2(d-1)$", fontsize=16)
plt.ylabel("$\log_{10}$(RMSE of mean)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/error_mean_dimension_SVGD.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean_dimension_SVGD.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()

fig2 = plt.figure(2)
for i in range(Ntrial):
    plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionFalse[:, i]), makerFalse[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionTrue[:, i]), makerTrue[case], alpha=0.2)
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='SVGD')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pSVGD')
plt.legend(fontsize=16)
plt.xlabel("$\log_2(d-1)$", fontsize=16)
plt.ylabel("$\log_{10}$(RMSE of variance)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/error_variance_dimension_SVGD.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance_dimension_SVGD.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()



#################################### WGF ###################

meanErrorL2normTrue = np.zeros((4, Ntrial, Niter))
varianceErrorL2normTrue = np.zeros((4, Ntrial, Niter))
meanErrorL2normFalse = np.zeros((4, Ntrial, Niter))
varianceErrorL2normFalse = np.zeros((4, Ntrial, Niter))
# meanErrorL2normSVGD = np.zeros((4, Ntrial, Niter))
# varianceErrorL2normSVGD = np.zeros((4, Ntrial, Niter))

labelTrue = ['pWGF d=81', 'pWGF d=289', 'pWGF d=1089', 'pWGF d=4225']
labelFalse = ['WGF d=81', 'WGF d=289', 'WGF d=1089', 'WGF d=4225']
fig1 = plt.figure(1)
fig2 = plt.figure(2)
for case in [0, 1, 2, 3]:

    d, N = dSet[case], NSet[case]

    for i in range(Ntrial):
        filename = "accuracy-" + str(d) + 'x' + str(d) + "/data/data_nDimensions_"+str((d+1)**2)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(True) + "_WGF_" + str(i+1) + ".p"

        print("case = ", case, "i = ", i)
        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
        varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
        meanErrorL2normTrue[case, i, :] = meanErrorL2normTrue_i
        varianceErrorL2normTrue[case, i, :] = varianceErrorL2normTrue_i

        filename = "accuracy-" + str(d) + 'x' + str(d) + "/data/data_nDimensions_"+str((d+1)**2)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(False) + "_WGF_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
        varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
        meanErrorL2normFalse[case, i, :] = meanErrorL2normFalse_i
        varianceErrorL2normFalse[case, i, :] = varianceErrorL2normFalse_i

    plt.figure(1)
    for i in range(Ntrial):
        plt.plot(np.log10(meanErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
        plt.plot(np.log10(meanErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
    plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normTrue[case,:,:]**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])

    plt.figure(2)
    for i in range(Ntrial):
        plt.plot(np.log10(varianceErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
        plt.plot(np.log10(varianceErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normFalse[case,:,:]**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
    plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normTrue[case,:,:]**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])

plt.figure(1)
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of mean)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_mean_WGF.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean_WGF.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()

plt.figure(2)
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of variance)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_variance_WGF.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance_WGF.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()


meanErrorDimensionTrue = meanErrorL2normTrue[:, :, -1]
meanErrorDimensionFalse = meanErrorL2normFalse[:, :, -1]
varianceErrorDimensionTrue = varianceErrorL2normTrue[:, :, -1]
varianceErrorDimensionFalse = varianceErrorL2normFalse[:, :, -1]

case = 0

fig1 = plt.figure(1)
for i in range(Ntrial):
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionFalse[:, i]), makerFalse[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionTrue[:, i]), makerTrue[case], alpha=0.2)

plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='WGF')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pWGF')
plt.figure(1)
plt.legend(fontsize=16)
plt.xlabel("$\log_2(d-1)$", fontsize=16)
plt.ylabel("$\log_{10}$(RMSE of mean)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/error_mean_dimension_WGF.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean_dimension_WGF.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()

fig2 = plt.figure(2)
for i in range(Ntrial):
    plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionFalse[:, i]), makerFalse[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionTrue[:, i]), makerTrue[case], alpha=0.2)
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='WGF')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pWGF')
plt.legend(fontsize=16)
plt.xlabel("$\log_2(d-1)$", fontsize=16)
plt.ylabel("$\log_{10}$(RMSE of variance)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/error_variance_dimension_WGF.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance_dimension_WGF.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()
