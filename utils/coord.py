import numpy as np


def coord_3d(X,dim):
    phi   = X[:,1]/dim[1] * np.pi     # phi
    theta = X[:,0]/dim[0] * 2 * np.pi         # theta
    R = np.stack([(np.sin(phi) * np.cos(theta)).T,(np.sin(phi) * np.sin(theta)).T,np.cos(phi).T], axis=1)

    return R

