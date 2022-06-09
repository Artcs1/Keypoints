import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from os import scandir, getcwd
from os.path import abspath,isfile

from scipy.spatial.transform import Rotation as R


def lvl_path(path):
    """
        count the number of / of an absolute path
        path : string -> absolute path
    """
    lvl = 0
    for c in path:
        if c =='/':
            lvl+=1
    return lvl

def listdir(ruta):
    """
        list directory path that contain the most number of /
        ruta: string -> absolute path
    """

    paths = []
    maxi = 0
    for (dirpath, _, _) in os.walk(ruta):

        maxi = max(maxi,lvl_path(dirpath))


    for (dirpath, _, _) in os.walk(ruta):
        if lvl_path(dirpath) == maxi:
            paths.append(dirpath)

    return paths

def FT_error(T1,T2):
    """
        translation error among 1x1 pair
        T1: matrix -> matched points X 3
        T2: matrix -> matched points X 3
    """
    result = np.diag(np.dot(T1,T2.T))/(np.linalg.norm(T1,axis=1)*np.linalg.norm(T2,axis = 1))
    return np.arccos(np.clip(result,-1,1))


def Tall_error(T1,T2):
    """
        translation error among all possible pairs
        T1: matrix -> feature points X 3
        T2: matrix -> feature points X 3
    """
    result = np.dot(T1,T2.T)/(np.dot(np.linalg.norm(T1,axis=1).reshape(-1,1),np.linalg.norm(T2,axis = 1).reshape(1,-1)))
    result = np.max(result,axis=1)
    return np.arccos(np.clip(result,-1,1))


def correct_matches_cor(x1, x2, R, T):
    """
        Repeatibility based on matched points
        x1: points of image1 -> number of matched points X 3
        x2: points of image2 -> number of matched points X 3
        R: ground truth rotation matrix    -> 3X3
        T: ground truth translation vector -> 3X1

    """
    x_1 = np.dot(x1.copy(), R.T) + T
    x_1 = x_1/np.linalg.norm(x_1,axis=1).reshape(-1,1)

    D = T_error(x_1,x2)

    return len(np.where(D<0.1)[0])

def correct_matches_all(p1, p2, R, T):
    """
        Repeatibility based on all points
        x1: points of image1 -> number of feature points X 3
        x2: points of image2 -> number of feature points X 3
        R: ground truth rotation matrix    -> 3X3
        T: ground truth translation vector -> 3X1
    """


    p_1 = np.dot(p1.copy(), R.T) + T
    p_1 = p_1/np.linalg.norm(p_1, axis=1).reshape(-1,1)

    D = Tall_error(p_1, p2)

    return len(np.where(D<0.1)[0])

def calc_entropy(p1, width=720,height=720,bins = 3, sigma = 1):

    """
        Calculates the entropy with bins among the sphere
        p1               : matrix -> features points x 3
        width and height : dimension of the sphere
        bins             : square(number of total bins)
        sigma            : constant value for the gaussian operator
    """

    mat_entropy = np.zeros((bins,bins))

    num_bins = bins**2

    mov_phi   = (np.pi/bins) *0.5
    mov_theta = (2*np.pi/bins) *0.5

    phi, theta = np.meshgrid(np.linspace(0, np.pi, num = bins, endpoint=False)+mov_phi, np.linspace(0, 2 * np.pi, num = bins, endpoint=False)+mov_theta)
    coordSph = np.stack([(np.sin(phi)*np.cos(theta)).T, (np.sin(phi)*np.sin(theta)).T, np.cos(phi).T],axis=2)
    x, y, z = coordSph[:,].T

    for keypoint in p1:
        for i in np.arange(bins):
            for j in np.arange(bins):
                bin_center = np.array([x[i,j], y[i,j], z[i,j]])
                dist = np.dot(keypoint,bin_center)/(np.linalg.norm(keypoint)*np.linalg.norm(bin_center))
                dist = np.arccos(np.clip(dist,-1,1))
                mat_entropy[i, j] += np.exp(-(dist**2)/(2*sigma**2))

    mat_entropy*= 1/(np.sum(mat_entropy)+1e-5)
    mat_entropy+=0.000001
    return np.sum(-mat_entropy*np.log(mat_entropy))


