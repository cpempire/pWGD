import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dSet = [16, 64, 256, 1024]
dPara = np.array(dSet)
NSet = [256, 256, 256, 256]
# NSet = [64, 64, 64, 64]
# NSet = [16, 16, 16, 16]

Ntrial = 10
Niter = 200
makerTrueBatch = ['r.--', 'k.--', 'm.--', 'c.--']
makerTrue = ['g.--', 'k.--', 'm.--', 'c.--']
makerFalse = ['b.--', 'r.--', 'y.--', 'b.--']
makerRMSETrueBatch = ['rd-', 'kd-', 'mo-', 'c*-']
makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']
makerRMSEFalse = ['bs-', 'r*-', 'y<-', 'b>-']

#################################### WGF ###################
meanErrorL2normTrueBatch = np.zeros((4, Ntrial, Niter))
varianceErrorL2normTrueBatch = np.zeros((4, Ntrial, Niter))
meanErrorL2normTrue = np.zeros((4, Ntrial, Niter))
varianceErrorL2normTrue = np.zeros((4, Ntrial, Niter))
meanErrorL2normFalse = np.zeros((4, Ntrial, Niter))
varianceErrorL2normFalse = np.zeros((4, Ntrial, Niter))
# meanErrorL2normSVGD = np.zeros((4, Ntrial, Niter))
# varianceErrorL2normSVGD = np.zeros((4, Ntrial, Niter))

labelTrue = ['pWGF d=17', 'pWGF d=65', 'pWGF d=257', 'pWGF d=1025']
labelFalse = ['WGF d=17', 'WGF d=65', 'WGF d=257', 'WGF d=1025']
fig1 = plt.figure(1)
fig2 = plt.figure(2)
for case in [0, 1, 2, 3]:

    d, N = dSet[case], NSet[case]

    for i in range(Ntrial):

        filename = "accuracy-" + str(d) + "/data/data_nDimensions_"+str(d+1)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(True) + "_WGF_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
        varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
        meanErrorL2normTrue[case, i, :] = meanErrorL2normTrue_i
        varianceErrorL2normTrue[case, i, :] = varianceErrorL2normTrue_i

        filename = "accuracy-" + str(d) + "/data/data_nDimensions_"+str(d+1)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(False) + "_WGF_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
        varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
        meanErrorL2normFalse[case, i, :] = meanErrorL2normFalse_i
        varianceErrorL2normFalse[case, i, :] = varianceErrorL2normFalse_i

        filename = "accuracy-" + str(d) + "/data/data_nDimensions_"+str(d+1)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(True) + "_WGF_batch5_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normTrueBatch_i = data_save["meanErrorL2norm"]
        varianceErrorL2normTrueBatch_i = data_save["varianceErrorL2norm"]
        meanErrorL2normTrueBatch[case, i, :] = meanErrorL2normTrueBatch_i
        varianceErrorL2normTrueBatch[case, i, :] = varianceErrorL2normTrueBatch_i

    plt.figure(1)
    for i in range(Ntrial):
        plt.plot(np.log10(meanErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
        plt.plot(np.log10(meanErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
        plt.plot(np.log10(meanErrorL2normTrueBatch[case, i, :]), makerTrue[case], alpha=0.2)
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

meanErrorDimensionTrueBatch = meanErrorL2normTrueBatch[:, :, -1]
meanErrorDimensionTrue = meanErrorL2normTrue[:, :, -1]
meanErrorDimensionFalse = meanErrorL2normFalse[:, :, -1]
varianceErrorDimensionTrueBatch = varianceErrorL2normTrueBatch[:, :, -1]
varianceErrorDimensionTrue = varianceErrorL2normTrue[:, :, -1]
varianceErrorDimensionFalse = varianceErrorL2normFalse[:, :, -1]

case = 0

fig1 = plt.figure(1)
for i in range(Ntrial):
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionFalse[:, i]), makerFalse[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionTrue[:, i]), makerTrue[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionTrueBatch[:, i]), makerTrueBatch[case], alpha=0.2)

plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='WGF')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pWGF')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionTrueBatch ** 2, axis=1))), makerRMSETrueBatch[case], linewidth=2,
         label='pWGF-batch')
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
    plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionTrueBatch[:, i]), makerTrueBatch[case], alpha=0.2)
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='WGF')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pWGF')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionTrueBatch ** 2, axis=1))), makerRMSETrueBatch[case], linewidth=2,
         label='pWGF-batch')
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


######################################## SVGD ##################################
Niter = 200
meanErrorL2normTrue = np.zeros((4, Ntrial, Niter))
varianceErrorL2normTrue = np.zeros((4, Ntrial, Niter))
meanErrorL2normFalse = np.zeros((4, Ntrial, Niter))
varianceErrorL2normFalse = np.zeros((4, Ntrial, Niter))
# meanErrorL2normSVGD = np.zeros((4, Ntrial, Niter))
# varianceErrorL2normSVGD = np.zeros((4, Ntrial, Niter))

labelTrue = ['pSVGD d=17', 'pSVGD d=65', 'pSVGD d=257', 'pSVGD d=1025']
labelFalse = ['SVGD d=17', 'SVGD d=65', 'SVGD d=257', 'SVGD d=1025']
fig1 = plt.figure(1)
fig2 = plt.figure(2)
for case in [0, 1, 2, 3]:

    d, N = dSet[case], NSet[case]

    for i in range(Ntrial):
        filename = "accuracy-" + str(d) + "/data/data_nDimensions_"+str(d+1)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(True) + "_SVGD_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
        varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
        meanErrorL2normTrue[case, i, :] = meanErrorL2normTrue_i
        varianceErrorL2normTrue[case, i, :] = varianceErrorL2normTrue_i

        filename = "accuracy-" + str(d) + "/data/data_nDimensions_"+str(d+1)+"_nCores_"+str(16)+"_nSamples_" + str(N) + "_isProjection_" + str(False) + "_SVGD_" + str(i+1) + ".p"

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