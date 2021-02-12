import numpy as np


def MMD(X,Y,bandwidth=1.):
    # MMD based on Gaussian kernel
    # X, Y are with shapes N*d and M*d
    N, d = X.shape
    M, _ = Y.shape
    h = bandwidth
    X_sum = np.sum(X**2,axis=1)
    Y_sum = np.sum(Y**2,axis=1)
    Dxx = -2*np.matmul(X, X.T)+X_sum.reshape([-1,1])+X_sum.reshape([1,-1])
    Dxy = -2*np.matmul(X, Y.T)+X_sum.reshape([-1,1])+Y_sum.reshape([1,-1])
    Dyy = -2*np.matmul(Y, Y.T)+Y_sum.reshape([-1,1])+Y_sum.reshape([1,-1])
    Kxx = np.exp(-Dxx/(2*h))
    Kxy = np.exp(-Dxy/(2*h))
    Kyy = np.exp(-Dyy/(2*h))
    res = 1./N**2*np.sum(Kxx.ravel())-2./(M*N)*np.sum(Kxy.ravel())+1./M**2*np.sum(Kyy.ravel())
    res = res*(2*np.pi*h)**(-d/2)
    return res

def calculate_BM_ibw(X, ibw_in, alpha, explore_ratio=1.1):
    N, d = X.shape
    Y = X+np.sqrt(2*alpha)*np.random.randn(N,d)
    X_sum = np.sum(X**2,axis=1)
    Dxx = -2*np.matmul(X, X.T)+X_sum.reshape([-1,1])+X_sum.reshape([1,-1])
    def get_obj_ibandw(ibw):
        Kxy = np.exp(-Dxx*0.5*ibw)
        dxKxy = np.matmul(Kxy,X)
        sumKxy = np.sum(Kxy,1).reshape([-1,1])
        xi = ibw*(dxKxy/sumKxy-X)
        X_new = X-alpha*xi
        res = MMD(X_new,Y,1.)
        return res
    obj_ibw_in = get_obj_ibandw(ibw_in)
    epsi = 1e-6
    grad_ibw_in = (get_obj_ibandw(ibw_in+epsi)-obj_ibw_in)/epsi
    if grad_ibw_in<0:
        ibw_1 = ibw_in*explore_ratio
    else:
        ibw_1 = ibw_in/explore_ratio
    obj_ibw_1 = get_obj_ibandw(ibw_1)
    slope_ibw = (obj_ibw_1-obj_ibw_in)/(ibw_1-ibw_in)
    ibw_2 = (ibw_in * slope_ibw - 0.5 * grad_ibw_in * (ibw_1 + ibw_in)) / (slope_ibw - grad_ibw_in)
    obj_ibw_2 = get_obj_ibandw(ibw_2)
    if not np.isnan(ibw_2) and ibw_2>0:
        if obj_ibw_1<obj_ibw_in:
            if obj_ibw_2<obj_ibw_1:
                ibw_out = ibw_2
            else:
                ibw_out = ibw_1
        else:
            if obj_ibw_2<obj_ibw_in:
                ibw_out = ibw_2
            else:
                ibw_out = ibw_in
    else:
        if obj_ibw_1<obj_ibw_in:
            ibw_out = ibw_1
        else:
            ibw_out = ibw_in
        
    return ibw_out

def partial_copy(vector, index):
    n = len(vector)
    new_vector = np.zeros(n)
    new_vector[index] = vector[index]
    return new_vector