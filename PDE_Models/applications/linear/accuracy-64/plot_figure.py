import pickle
import numpy as np
import matplotlib.pyplot as plt

dSet = [17, 65, 257, 1024]
dPara = np.array(dSet)
# NSet = [256, 256, 256, 256]
# NSet = [64, 64, 64, 64]
NSet = [16, 16, 16, 16]

Ntrial = 10
Niter = 200

# makerTrue = ['g.--', 'k.--', 'm.--', 'c.--']
# makerFalse = ['b.--', 'r.--', 'y.--', 'b.--']
# makerFalse_WGF = [ 'm.--','c.--', 'k.--', 'g.--']
makerTrue = ['g', 'k', 'm', 'c']
makerFalse = ['b', 'r', 'y', 'b']
makerFalse_WGF = [ 'm','c', 'k', 'g']
makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']
makerRMSEFalse = ['bs-', 'r*-', 'y<-', 'b>-']
makerRMSEFalse_WGF = ['mo-','c*-', 'bs-', 'r*-']
lineRMSETrue = ['g-', 'k-', 'm-', 'c-']
lineRMSEFalse = ['b-', 'r-', 'y-', 'b-']
lineRMSEFalse_WGF = ['m-','c-', 'b-', 'r-']

meanErrorL2normTrue = np.zeros((4, Ntrial, Niter))
varianceErrorL2normTrue = np.zeros((4, Ntrial, Niter))
meanErrorL2normFalse = np.zeros((4, Ntrial, Niter))
varianceErrorL2normFalse = np.zeros((4, Ntrial, Niter))
meanErrorL2normFalse_WGF = np.zeros((4, Ntrial, Niter))
varianceErrorL2normFalse_WGF = np.zeros((4, Ntrial, Niter))
meanErrorL2normSVGD = np.zeros((4, Ntrial, Niter))
varianceErrorL2normSVGD = np.zeros((4, Ntrial, Niter))
labelTrue = ['pSVGD d=17', 'pSVGD d=65', 'pSVGD d=257', 'pSVGD d=1025']
labelFalse = ['SVGD d=17', 'SVGD d=65', 'SVGD d=257', 'SVGD d=1025']
labelFalse_WGF = ['WGF d=17', 'WGF d=65', 'WGF d=257', 'WGF d=1025']
fig1 = plt.figure(1)
fig2 = plt.figure(2)

case = 1
d, N = dSet[case], NSet[case]

for i in range(Ntrial):
    filename = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_False_SVGD_{}.p'.format(d,N,i+1)
    data_save = pickle.load(open(filename, 'rb'))
    meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
    varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
    meanErrorL2normFalse[case, i, :] = meanErrorL2normFalse_i
    varianceErrorL2normFalse[case, i, :] = varianceErrorL2normFalse_i
    filename = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_SVGD_{}.p'.format(d,N,i+1)
    data_save = pickle.load(open(filename, 'rb'))
    meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
    varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
    meanErrorL2normTrue[case, i, :] = meanErrorL2normTrue_i
    varianceErrorL2normTrue[case, i, :] = varianceErrorL2normTrue_i
    filename = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_False_WGF_{}.p'.format(d,N,i+1)
    data_save = pickle.load(open(filename, 'rb'))
    meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
    varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
    meanErrorL2normFalse_WGF[case, i, :] = meanErrorL2normFalse_i
    varianceErrorL2normFalse_WGF[case, i, :] = varianceErrorL2normFalse_i

fig1=plt.figure(1)

interval = 10
iter_num = 200
iters = np.arange(iter_num)

# i = 1
# print(meanErrorL2normFalse[case, i, :])

# print(iters[0:200:5])
# print(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0)))[0:200:5])
for i in range(Ntrial):
    plt.plot(np.log10(meanErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
    plt.plot(np.log10(meanErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
    plt.plot(np.log10(meanErrorL2normFalse_WGF[case, i, :]), makerFalse_WGF[case], alpha=0.2)
plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0))), lineRMSEFalse[case], linewidth=3)
plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normTrue[case,:,:]**2, axis=0))), lineRMSETrue[case], linewidth=3)
plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normFalse_WGF[case,:,:]**2, axis=0))), lineRMSEFalse_WGF[case], linewidth=3)
plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0)))[0:iter_num:interval], makerRMSEFalse[case], linewidth=3, label=labelFalse[case])
plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(meanErrorL2normTrue[case,:,:]**2, axis=0)))[0:iter_num:interval], makerRMSETrue[case], linewidth=3, label=labelTrue[case])
plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(meanErrorL2normFalse_WGF[case,:,:]**2, axis=0)))[0:iter_num:interval], makerRMSEFalse_WGF[case], linewidth=3, label=labelFalse_WGF[case])
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of mean)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_mean_"+"SVGD.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean_"+"SVGD.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()


fig2 = plt.figure(2)
case = 1

d, N = dSet[case], NSet[case]

plt.figure(2)
for i in range(Ntrial):
    plt.plot(np.log10(varianceErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.1)
    plt.plot(np.log10(varianceErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.1)
    plt.plot(np.log10(varianceErrorL2normFalse_WGF[case, i, :]), makerFalse_WGF[case], alpha=0.1)
plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normFalse[case,:,:]**2, axis=0))), lineRMSEFalse[case], linewidth=3)
plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normTrue[case,:,:]**2, axis=0))), lineRMSETrue[case], linewidth=3)
plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normFalse_WGF[case,:,:]**2, axis=0))), lineRMSEFalse_WGF[case], linewidth=3)

plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(varianceErrorL2normFalse[case,:,:]**2, axis=0)))[0:iter_num:interval], makerRMSEFalse[case], linewidth=3, label=labelFalse[case])
plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(varianceErrorL2normTrue[case,:,:]**2, axis=0)))[0:iter_num:interval], makerRMSETrue[case], linewidth=3, label=labelTrue[case])
plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(varianceErrorL2normFalse_WGF[case,:,:]**2, axis=0)))[0:iter_num:interval], makerRMSEFalse_WGF[case], linewidth=3, label=labelFalse_WGF[case])
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of variance)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_variance_" + "SVGD.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance_" + "SVGD.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()