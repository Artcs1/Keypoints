import numpy as np


def computeEssentialMatrix(x1, x2):
    return calc_ematrix(x1,x2), range(len(x1))

# This function estimates the Fundamental matrix from a pair of 3D point correspondences rejecting outliers via RANSAC
#
# @param x1 is a set of n 3D points corresponding to x2 represented as a Numpy array
# @param x2 is a set of n 3D points corresponding to x1 represented as a Numpy array
# @param p is the minimum percentage of inliers
# @param k is the size of the data to estimate an initial model
# @param e is the individual error for assuming a datum is inlier
# @param m is the maximum number of iterations
# WARNING: initial tests consider p = .85 and e = 1e-3
def computeEssentialMatrixByRANSAC(x1, x2, w = None, p = 0.1, k = 8, e = 1e-1, m = 1000):
    it = 0
    betterIdx = []
    bestIdx = []
    besterror = np.inf
    minimumAcceptable = p * x1.shape[0]
    bestmodel = None
    bettermodel = None
    bestinliersx1 = None
    bestinliersx2 = None
    betterinliersx1 = None
    betterinliersx2 = None
    bestNumberOfInliers = -np.inf
    while(it != m):
        fitIdx, testIdx = __partition(len(x1), k)
        fitx1 = x1[fitIdx]; fitx2 = x2[fitIdx]
        testx1 = x1[testIdx]; testx2 = x2[testIdx]
        fitmodel = calc_ematrix(fitx1, fitx2)
        errors = symmetricProjectedDistance(fitmodel, testx1, testx2)

        auxIdx = np.where(errors <= e)[0]
        inliersx1 = testx1[auxIdx]
        inliersx2 = testx2[auxIdx]

        testIdx = testIdx[auxIdx]
        #print(minimumAcceptable)
        #print(len(inliersx1)+len(fitx1))
        if (len(inliersx1)+len(fitx1)) >= minimumAcceptable:
            betterIdx = np.concatenate((testIdx,fitIdx))
            betterinliersx1 = np.vstack((inliersx1,fitx1))
            betterinliersx2 = np.vstack((inliersx2,fitx2))
            bettermodel = calc_ematrix(betterinliersx1, betterinliersx2)
            bettererror = np.mean(symmetricProjectedDistance(bettermodel, betterinliersx1, betterinliersx2))
            if (len(inliersx1)+len(fitx1)) > bestNumberOfInliers:
                bestNumberOfInliers = (len(inliersx1)+len(fitx1))
                bestIdx = betterIdx
                bestmodel = bettermodel
                besterror = bettererror
                bestinliersx1 = betterinliersx1
                bestinliersx2 = betterinliersx2
	    		# break
        it += 1

    return bestmodel, np.asarray(sorted(bestIdx))


# This function computes the Sampson distance between two 3D points given a Fundamental matrix
#
# @param x1 is a 3D point corresponding to x2 represented as a Numpy array
# @param x2 is a 3D point corresponding to x1 represented as a Numpy array
# @param F is an estimate of the Fundamental matrix represented as a Numpy matrix
def __sampsonDistance(F, x1, x2):
	Fx1  = x1.dot(F.T)
	x2Fx1 = np.einsum('ij,ij->i', x2, Fx1)**2
	sumFx1 = sum(Fx1**2,axis=1)
	sumFx2 = sum((x2.dot(F))**2,axis=1)
	return x2Fx1 / sumFx1 + x2Fx1 / sumFx2



# This function computes the Projected distance (10.1109/ICCVW.2011.6130266) between two 3D points given a Essential matrix
#
# @param x1 is a 3D point corresponding to x2 represented as a Numpy array
# @param x2 is a 3D point corresponding to x1 represented as a Numpy array
# @param E is an estimate of the Essential matrix represented as a Numpy matrix
def __projectedDistance(E, x1, x2):
	Ex1 = x1.dot(E.T)
	return np.abs(np.einsum('ij,ij->i',x2,Ex1))/np.linalg.norm(Ex1, axis=1)



# This function computes the Projected distance (10.1109/ICCVW.2011.6130266) between two 3D points given a Essential matrix
#
# @param x1 is a 3D point corresponding to x2 represented as a Numpy array
# @param x2 is a 3D point corresponding to x1 represented as a Numpy array
# @param E is an estimate of the Essential matrix represented as a Numpy matrix
def symmetricProjectedDistance(E, x1, x2):
	return (__projectedDistance(E,x1,x2) + __projectedDistance(E.T,x2,x1))/2.



# This function makes a partition of sizes k and n-k pairs of points 3D
#
# @param x1 is a set of n 3D points corresponding to x2 represented as a Numpy array
# @param x2 is a set of n 3D points corresponding to x1 represented as a Numpy array
# @param k is the size of the first subset; n-k is the size of the second subset
def __partition(n, k):
	idx = np.arange(n)
	np.random.shuffle(idx)
	a, b = np.split(idx, [k])
	return a, b

# This function internally estimate the Fundamental matrix from a pair of 3D point correspondences
#
# @param x1 is a set of n 3D points corresponding to x2 represented as a Numpy array
# @param x2 is a set of n 3D points corresponding to x1 represented as a Numpy array

def calc_ematrix(x1,x2,w=None):

    A = np.vstack((x1[:,0] * x2[:,0], x1[:,0] * x2[:,1], x1[:,0] * x2[:,2],
				x1[:,1] * x2[:,0], x1[:,1] * x2[:,1], x1[:,1] * x2[:,2],
				x1[:,2] * x2[:,0], x1[:,2] * x2[:,1], x1[:,2] * x2[:,2])).T


    if w is None:
        A = A.T@A
    else:
        W = np.diag(w)
        A = A.T@W@A

    UA, SA, VA = np.linalg.svd(A)
    f = VA.T[:,-1]
    F = f.reshape((3,3)).T

    UF, SF, VF = np.linalg.svd(F)
    SF = np.diag([1,1,0])

    F = UF@SF@VF

    return F

def t_error(T1,T2):
    result = np.dot(T1.T,T2.T)/(np.linalg.norm(T1)*np.linalg.norm(T2))
    return np.arccos(np.clip(result,-1,1))

def r_error(R1,R2):
    result = (np.trace(R1.T@R2)-1)/2
    return np.arccos(np.clip(result,-1,1))

def decomposeE(F):

    UF, SF, VF = np.linalg.svd(F)

    if np.linalg.det(UF)<=0:
        UF[:,2] = -UF[:,2]
    if np.linalg.det(VF.T)<=0:
        VF[2,:] = -VF[2,:]

    W = np.zeros((3,3))
    W[0,1] = 1
    W[1,0] = -1
    W[2,2] = 1
    R1 = UF@W@VF

    W = np.zeros((3,3))
    W[0,1] = -1
    W[1,0] = 1
    W[2,2] = 1
    R2 = UF@W@VF

    return R1,R2,UF[:,2],-UF[:,2]

def choose_rt(R1,R2,T1,T2,x1,x2):


    groups = [(R1,T1),(R1,T2),(R2,T1),(R2,T2)]
    votes = np.zeros(4)

    for i in range(x1.shape[0]):
        j = 0
        for r,t in groups:
            f1 = x2[i,:]
            f2 = -r@x1[i,:]

            A = np.stack((f1,f2),axis = -1)

            equiA = A.T@A
            equit = A.T@t

            b,a = np.linalg.solve(equiA,equit)
            if a > 0 and b > 0:
                votes[j] = votes[j]+1

            j = j +1

    ind = np.argmax(votes)
    return groups[ind]



