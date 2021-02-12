import pickle
import numpy as np
import matplotlib.pyplot as plt

dSet = [17, 64, 256, 1024]
dPara = np.array(dSet)
# NSet = [256, 256, 256, 256]
# NSet = [64, 64, 64, 64]
NSet = [16, 16, 16, 16]

Ntrial = 10
Niter = 200

fontsize = 12

colors = ['g', 'k', 'm', 'c', 'b', 'y', 'r','g', 'k', 'm', 'c', 'b', 'y', 'r']
markers = ['x-', 'd-', 'o-', '*-', 's-', '<-', '*-','x-', 'd-', 'o-', '*-', 's-', '<-', '*-']

case = 0
d, N = dSet[case], NSet[case]

SVGD = {}
SVGD['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
SVGD['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
SVGD['label'] = 'SVGD d=17'
SVGD['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_False_SVGD'.format(d,N)

pSVGD = {}
pSVGD['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pSVGD['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
pSVGD['label'] = 'pSVGD d=17'
pSVGD['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_SVGD_prior'.format(d,N)

pSVGD_post = {}
pSVGD_post['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pSVGD_post['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
pSVGD_post['label'] = 'pSVGD-post d=17'
pSVGD_post['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_SVGD'.format(d,N)

WGF = {}
WGF['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
WGF['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
WGF['label'] = 'WGF d=17'
WGF['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_False_WGF'.format(d,N)

WGF_BM = {}
WGF_BM['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
WGF_BM['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
WGF_BM['label'] = 'WGF-BM d=17'
WGF_BM['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_False_WGF_BM'.format(d,N)

pWGF = {}
pWGF['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF['label'] = 'pWGF d=17'
pWGF['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF'.format(d,N)

pWGF_BM = {}
pWGF_BM['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_BM['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_BM['label'] = 'pWGF-BM d=17'
pWGF_BM['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_BM'.format(d,N)

pWGF_post = {}
pWGF_post['meanErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_post['varianceErrorL2norm'] = np.zeros((Ntrial, Niter))
pWGF_post['label'] = 'pWGF-post d=17'
pWGF_post['filename'] = './data/data_nDimensions_{}_nCores_1_nSamples_{}_isProjection_True_WGF_post'.format(d,N)


algorithms = [SVGD, pSVGD, WGF, pWGF, WGF_BM, pWGF_BM]


for i in range(Ntrial):
    for algo in algorithms:
        filename = '{}_{}.p'.format(algo['filename'],i+1)
        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2norm = data_save["meanErrorL2norm"]
        varianceErrorL2norm = data_save["varianceErrorL2norm"]
        # print(algo['meanErrorL2norm'].shape)
        algo['meanErrorL2norm'][i, :] = meanErrorL2norm
        algo['varianceErrorL2norm'][i, :] = varianceErrorL2norm

    
fig1=plt.figure(1)

interval = 10
iter_num = 200
iters = np.arange(iter_num)

# print(iters[0:200:5])
# print(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0)))[0:200:5])
for (j, algo) in enumerate(algorithms):
    for i in range(Ntrial):
        plt.plot(np.log10(algo['meanErrorL2norm'][i, :]), colors[j], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(algo['meanErrorL2norm'][:, :]**2, axis=0))), colors[j], linewidth=3)
    plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(algo['meanErrorL2norm'][:, :]**2, axis=0)))[0:iter_num:interval], colors[j]+markers[j], linewidth=3, label=algo['label'])

plt.legend(fontsize=fontsize)
plt.xlabel("# iterations", fontsize=fontsize)
plt.ylabel("Log10(RMSE of mean)", fontsize=fontsize)

plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.tick_params(axis='both', which='minor', labelsize=fontsize)

filename = "figure/error_mean.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()


fig2 = plt.figure(2)

for (j, algo) in enumerate(algorithms):
    for i in range(Ntrial):
        plt.plot(np.log10(algo['varianceErrorL2norm'][i, :]), colors[j], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(algo['varianceErrorL2norm'][:, :]**2, axis=0))), colors[j], linewidth=3)
    plt.plot(iters[0:iter_num:interval],np.log10(np.sqrt(np.mean(algo['varianceErrorL2norm'][:, :]**2, axis=0)))[0:iter_num:interval], colors[j]+markers[j], linewidth=3, label=algo['label'])

plt.legend(fontsize=fontsize)
plt.xlabel("# iterations", fontsize=fontsize)
plt.ylabel("Log10(RMSE of variance)", fontsize=fontsize)

plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.tick_params(axis='both', which='minor', labelsize=fontsize)

filename = "figure/error_variance.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()