"""
    PROGRAMA PARA CALCULAR KEYPOINTS DE DOS IMAGENES 512x1024, ASI MISMO DE SUS CORRESPONDENCIAS
    POR UN KNN BILATERAL

"""

import time
import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images, unresample
from spherical_distortion.util import *
import matplotlib.pyplot as plt
from skimage import io
import os
import cv2

import sys
import pandas as pd
import numpy as np
import _spherical_distortion_ext._mesh as _mesh
import argparse

from random import sample
import imageio
from scipy.spatial.transform import Rotation as Rot

from utils.coord    import coord_3d
from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *

from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm

sys.path.append(os.getcwd()+'/SPHORB-master')

import build1.sphorb_cpp as sphorb

def sort_key(pts1, pts2, desc1, desc2, points):

    ind1 = np.argsort(pts1[:,2].numpy(),axis = 0)[::-1]
    ind2 = np.argsort(pts2[:,2].numpy(),axis = 0)[::-1]

    max1 = np.min([points,ind1.shape[0]])
    max2 = np.min([points,ind2.shape[0]])

    ind1 = ind1[:max1]
    ind2 = ind2[:max2]

    pts1 = pts1[ind1.copy(),:]
    pts2 = pts2[ind2.copy(),:]

    desc1 = desc1[:,ind1.copy()]
    desc2 = desc2[:,ind2.copy()]

    pts1 = np.concatenate((pts1[:,:2], np.ones((pts1.shape[0],1))), axis = 1 )
    pts2 = np.concatenate((pts2[:,:2], np.ones((pts2.shape[0],1))), axis = 1 )

    desc1 = np.transpose(desc1,[1,0]).numpy()
    desc2 = np.transpose(desc2,[1,0]).numpy()

    return pts1, pts2, desc1, desc2

def matched_points(pts1, pts2, desc1, desc2, opt, args_opt, match='ratio'):

    if opt[-1] == 'p':
        porce = int(opt[:-1])
        n_key = int(porce/100 * pts1.shape[0])
    else:
        n_key = int(opt)

    s_pts1  = pts1.copy()[:n_key,:]
    s_pts2  = pts2.copy()[:n_key,:]
    s_desc1 = desc1.copy().astype('float32')[:n_key,:]
    s_desc2 = desc2.copy().astype('float32')[:n_key,:]

    if  'orb' in args_opt:
        s_desc1 = s_desc1.astype(np.uint8)
        s_desc2 = s_desc2.astype(np.uint8)

    if match == '2-cross':
        if 'orb' in args_opt:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, True)
        matches = bf.match(s_desc1, s_desc2)
    elif match == 'ratio':
        if 'orb' in args_opt:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches = bf.knnMatch(s_desc1,s_desc2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = good

    M = np.zeros((2,len(matches)))
    for ind, match in zip(np.arange(len(matches)),matches):
        M[0,ind] = match.queryIdx
        M[1,ind] = match.trainIdx

    num_M = M.shape[1]

    return s_pts1, s_pts2, s_pts1[M[0,:].astype(int),:3], s_pts2[M[1,:].astype(int),:3]

def get_error(x1, x2, Rx, Tx):

    S = computeEssentialMatrixByRANSAC(x1, x2)
    I = S[1]
    I = I.astype(np.int64)

    x1 = x1[I,:]
    x2 = x2[I,:]

    F = calc_ematrix(x1,x2)


    R1,R2,T1,T2 = decomposeE(F)

    R,T = choose_rt(R1,R2,T1,T2,x1,x2)

    R_error = r_error(Rx,R)
    T_error = t_error(Tx,T)

    return R_error, T_error

def get_descriptor(descriptor):
    if descriptor == 'sphorb':
        return 'sphorb', 'erp', 640
    elif descriptor == 'sift':
        return 'sift', 'erp', 512
    elif descriptor == 'tsift':
        return 'sift', 'tangent', 512
    elif descriptor == 'orb':
        return 'orb', 'erp', 512
    elif descriptor == 'torb':
        return 'orb', 'tangent', 512
    elif descriptor == 'spoint':
        return 'superpoint', 'erp', 512
    elif descriptor == 'tspoint':
        return 'superpoint', 'tangent', 512


def AUC(ROT, TRA, MET, L):

    RAUC  = np.zeros(len(L))
    TAUC  = np.zeros(len(L))

    for index, t in enumerate(L):
        ids = np.where(ROT<np.radians(t))[0]
        RAUC[index] = len(ids)/len(ROT)

    for index, t in enumerate(L):
        ids = np.where(TRA<np.radians(t))[0]
        TAUC[index] = len(ids)/len(TRA)

    return RAUC, TAUC, np.array(MET)

def get_data(DATAS):
    if len(DATAS) == 1:
        data = DATAS[0]
    elif set(['Urban1','Urban2','Urban3','Urban4']) == set(DATAS):
        data = 'Outdoor'
    elif set(['Realistic','Interior1','Interior2','Room','Classroom']) == set(DATAS):
        data = 'Indoor'
    elif set(['Urban1_R','Urban2_R','Urban3_R','Urban4_R','Realistic_R','Interior1_R','Interior2_R','Room_R','Classroom_R']) == set(DATAS):
        data = 'OnlyRot'
    elif set(['Urban1_T','Urban2_T','Urban3_T','Urban4_T','Realistic_T','Interior1_T','Interior2_T','Room_T','Classroom_T']) == set(DATAS):
        data = 'OnlyTra'
    else:
        data = ''
        for DA in DATAS:
            data+=DA

    return data


def get_kd(array):

    array = np.array(array)
    delimiter = int(array[-1])
    A = array[:-1]
    K = A[:delimiter].reshape(-1,3)
    D = A[delimiter:].reshape(-1,32)
    return K,D


def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 12000)
    parser.add_argument('--match', default="ratio")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="None")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--datas'      , nargs='+')
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()


    DATAS       = args.datas
    DESCRIPTORS = args.descriptors
    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    NUM = 0
    R_ERROR, T_ERROR = [], []
    for i in range(len(DESCRIPTORS)):
        R_ERROR.append([])
        T_ERROR.append([])

    if args.g_metrics == "False":
        METRICS = np.zeros((len(DESCRIPTORS),2))
        metrics = ['Matched','Keypoint']
    else:
        METRICS = np.zeros((len(DESCRIPTORS),7))
        metrics = ['Matched','Keypoint','Pmr','Pr','R','Ms','E']

    data = get_data(DATAS)
    TIMES = []
    for data in DATAS:

        mypath = os.path.join('data',data)
        paths  = [os.path.join(os.getcwd(),'data',data,f) for f in listdir('data/'+data) if isdir(join(mypath, f))]
        NUM = NUM + len(paths)
        
        std = []

        for path in tqdm(paths):

            for indicador, descriptor in enumerate(DESCRIPTORS):



                try:

                    opt, mode, sphered = get_descriptor(descriptor)

                    base_order = 1  # Base sphere resolution
                    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                    scale_factor = 1.0  # How much to scale input equirectangular image by
                    save_ply = False  # Whether to save the PLY visualizations too
                    dim = np.array([2*sphered, sphered])

                    path_o = path + '/O.png'
                    path_r = path + '/R.png'

                    if opt != 'sphorb':

                        # ----------------------------------------------
                        # Compute necessary data
                        # ----------------------------------------------
                        # 80 baricenter points
                        corners = tangent_image_corners(base_order, sample_order)

                        pts1, desc1 = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)

                        pts2, desc2 = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)

                        pts1, pts2, desc1, desc2 = sort_key(pts1, pts2, desc1, desc2, args.points)


                    else:
                        
                        #os.system('mogrify -format jpg '+path+'/*.png')
                        os.chdir('SPHORB-master/')

                        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
                        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
                        os.chdir('../')


                    depth = np.load("results/depth/"+data+".npy")

                    if len(pts1.shape) == 1:
                        pts1 = pts1.reshape(1,-1)

                    if len(pts2.shape) == 1:
                        pts2 = pts2.reshape(1,-1)
                    Rx = np.load(path+"/R.npy")
                    Tx = np.load(path+"/T.npy")

                    for indice, option in enumerate(['100p']):

                        if pts1.shape[0] > 0 or pts2.shape[0] >0:

                            s_pts1, s_pts2, x1, x2 = matched_points(pts1, pts2, desc1, desc2, option, opt, args.match)

                            z_d = depth[(x1[:,1]*512/sphered).astype('int')%512,(x1[:,0]*512/sphered).astype('int')%1024]
                            z_d2 = depth[(s_pts1[:,1]*512/sphered).astype('int')%512,(s_pts1[:,0]*512/sphered).astype('int')%1024]

                            x1,x2 = coord_3d(x1, dim), coord_3d(x2, dim)
                            x2_ = (x1.copy()*z_d.reshape(-1,1))@Rx.T + Tx
                            x2_ = x2_/np.linalg.norm(x2_,axis=1).reshape(-1,1)

                            s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)
                            s_pts2_ = (s_pts1.copy()*z_d2.reshape(-1,1))@Rx.T + Tx
                            s_pts2_ = s_pts2_/np.linalg.norm(s_pts2_,axis=1).reshape(-1,1)


                            if args.g_metrics == "True":

                                d2  = Tall_error(s_pts2, s_pts2_)
                                d   = FT_error(x2,x2_)
                                
                                EPS = 1e-5
                                pmr = x1.shape[0]/(s_pts1.shape[0]+EPS)
                                Pr  = np.sum(d<0.1)/(x1.shape[0]+EPS)
                                r   = np.sum(d<0.1)/(np.sum(d2<0.01)+EPS)
                                ms  = np.sum(d<0.1)/(s_pts1.shape[0]+EPS)
                                En  = calc_entropy(x1)


                            if x1.shape[0] < 8:
                                R_error, T_error = 3.14, 3.14
                            else:
                                inicio = time.time()
                                #time.sleep(1)
                                if args.solver   == 'None':
                                    E, cam = get_cam_pose_by_ransac_8pa(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                                elif args.solver == 'SK':
                                    E, can = get_cam_pose_by_ransac_opt_SK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                                elif args.solver == 'GSM':
                                    E, can = get_cam_pose_by_ransac_GSM(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                                elif args.solver == 'GSM_wRT':
                                    E, can = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                                elif args.solver == 'GSM_SK':
                                    E, can = get_cam_pose_by_ransac_GSM_const_wSK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)

                                fin = time.time()
                                TIMES.append(fin-inicio)
                                R1_,R2_,T1_,T2_ = decomposeE(E.T)
                                R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)

                                #R_error, T_error = get_error(x1, x2, Rx, Tx)
                                R_error, T_error = r_error(Rx,R_), t_error(Tx,T_)

                            R_ERROR[indicador].append(R_error)
                            T_ERROR[indicador].append(T_error)

                            if args.g_metrics == "False":
                                METRICS[indicador,:] = METRICS[indicador,:] + [x1.shape[0], (s_pts1.shape[0]+s_pts2.shape[1])/2]
                            else:
                                METRICS[indicador,:] = METRICS[indicador,:] + [x1.shape[0], (s_pts1.shape[0]+s_pts2.shape[1])/2,pmr,Pr,r,ms,En]

                            std.append(x1.shape[0])
                except:
                    print("Unexpected error")

        #print(np.array(std).std())



        print(np.mean(np.array(TIMES)))
    print('ALL:')
    print(np.mean(np.array(TIMES)))
    N_ARRAY = np.zeros((len(DESCRIPTORS),1))
    for indice, _ in enumerate(DESCRIPTORS):
        N_ARRAY[indice,0] = len(R_ERROR[indice])

    METRICS = METRICS/N_ARRAY
    METRICS = METRICS*100

    for file in ['100p']:


        data = get_data(DATAS)
        print("Results in data: "+ data +" with " + file + " of keypoints")

        P = DESCRIPTORS
        L = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        ROT, TRA, MET = np.zeros((len(P),len(L))), np.zeros((len(P),len(L))), np.zeros((len(P),METRICS.shape[1]))

        for i in range(len(P)):
            ROT[i,:], TRA[i,:], MET[i,:] = AUC(R_ERROR[i], T_ERROR[i], METRICS[i], L)

        mat_R = {}
        mat_T = {}
        mat_M = {}

        for ind, l in enumerate(L):
            mat_R[l]=ROT[:,ind]*100
            mat_T[l]=TRA[:,ind]*100

        for ind, m in enumerate(metrics):
            mat_M[m]=MET[:,ind]*100

        ROTATION = pd.DataFrame(mat_R, index=P)
        os.system('mkdir -p '+'results/metrics/'+data+'_'+file+'_'+args.inliers+'_'+args.solver)
        ROTATION.to_csv('results/metrics/'+data+'_'+file+'_'+args.inliers+'_'+args.solver+'/rotation.txt',float_format='%.2f')
        TRANSLATION = pd.DataFrame(mat_T, index=P)
        TRANSLATION.to_csv('results/metrics/'+data+'_'+file+'_'+args.inliers+'_'+args.solver+'/translation.txt', float_format='%.2f')
        METRICS = pd.DataFrame(mat_M, index=P)
        METRICS.to_csv('results/metrics/'+data+'_'+file+'_'+args.inliers+'_'+args.solver+'/metrics.txt', float_format='%.3f')

        print('ROTATION')
        print(ROT)
        print('TRANSLATION')
        print(TRA)
        print('GENERIC METRICS')
        print(MET)

#        fig, ax = plt.subplots(1,figsize=(8,6))
#
#        for i in range(len(P)):
#            ax.plot(L,ROT[i,:],'-o',linewidth=2,label = P[i])
#
#        plt.xlabel('degree')
#        plt.ylabel('accuracy')
#        ax.legend(loc="lower right")
#
#        plt.savefig('results/metrics/'+data+'_'+file+'_'+args.inliers+'_'+args.solver+'/Rotation.png')
#
#        plt.clf()
#
#
#        fig, ax = plt.subplots(1,figsize=(8,6))
#
#        for i in range(len(P)):
#            ax.plot(L,TRA[i,:],'-o',linewidth=2,label = P[i])
#
#        plt.xlabel('degree')
#        plt.ylabel('accuracy')
#        ax.legend(loc="lower right")
#
#        plt.savefig('results/metrics/'+data+'_'+file+'_'+args.inliers+'_'+args.solver+'/Translation.png')
#


if __name__ == '__main__':
    main()




