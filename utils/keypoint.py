import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images, unresample
from spherical_distortion.util import *
import matplotlib.pyplot as plt
from skimage import io
import os

import numpy as np
import _spherical_distortion_ext._mesh as _mesh
import argparse

import utils.superpoint.magic_sp.superpoint as magic_sp
import utils.superpoint.train_sp.superpoint as train_sp

def process_img(img):
    """ Process a image transposing it and convert to grayscale format, Then normalize

    img: 3 x H x W

    """
    img = np.transpose(img.numpy(),[1,2,0])
    H, W = img.shape[0], img.shape[1]
    grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #interp = cv2.INTER_AREA
    #grayim = cv2.resize(grayim, (160, 120), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

def computes_superpoint_keypoints(img, opt, nms_dist=4, conf_thresh = 0.015, nn_thresh =0.7, cuda = True):

    #if opt.enswith('.pth'):
    #fe = magic_sp.SuperPointFrontend(weights_path = 'models/superpoint_v1.pth', nms_dist= nms_dist, conf_thresh= conf_thresh, nn_thresh= nn_thresh,cuda= cuda )

    #else:

    fe = train_sp.SuperPointFrontend(weights_path = 'utils/models/superpoint-trained.pth.tar', nms_dist= nms_dist, conf_thresh = conf_thresh, nn_thresh= nn_thresh, cuda = cuda )



    pts, desc, heatmap = fe.run(img)
    kpt_details = np.zeros((pts.shape[1],4))
    kpt_details[:,0] = pts[0,:]
    kpt_details[:,1] = pts[1,:]
    kpt_details[:,2] = pts[2,:]
    kpt_details[:,3] = pts[2,:]
    if pts.shape[1] != 0:
        desc = np.transpose(desc, [1,0])
        return torch.from_numpy(kpt_details), torch.from_numpy(desc)
    return None

def format_keypoints(keypoints, desc):
    """
    Formatear puntos de interes y descriptores de opencv para su posterior tratamiento

    """
    coords = torch.tensor([kp.pt for kp in keypoints])
    responsex = torch.tensor([kp.response for kp in keypoints])
    responsey = torch.tensor([kp.response for kp in keypoints])
    desc = torch.from_numpy(desc)
    return torch.cat((coords, responsex.unsqueeze(1), responsey.unsqueeze(1)), -1), desc

def computes_orb_keypoints(img):

    img = torch2numpy(img.byte())

    # Initialize OpenCV ORB detector
    orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=10000)

    keypoints, desc = orb.detectAndCompute(img, None)

    # Keypoints is a list of lenght N, desc is N x 128
    if len(keypoints) > 0:
        return format_keypoints(keypoints, desc)
    return None

def computes_surf_keypoints(img):

    img = torch2numpy(img.byte())

    # Initialize OpenCV ORB detector
    surf = cv2.xfeatures2d.SURF_create(nfeatures=10000)

    keypoints, desc = surf.detectAndCompute(img, None)

    # Keypoints is a list of lenght N, desc is N x 128
    if len(keypoints) > 0:
        return format_keypoints(keypoints, desc)
    return None


def computes_sift_keypoints(img):

    img = torch2numpy(img.byte())

    # Initialize OpenCV ORB detector
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000)

    keypoints, desc = sift.detectAndCompute(img, None)

    # Keypoints is a list of lenght N, desc is N x 128
    if len(keypoints) > 0:
        return format_keypoints(keypoints, desc)
    return None



def keypoint_tangent_images(tex_image, base_order, sample_order, image_shape, opt = 'superpoint', crop_degree=0):
    """
    Extracts only the visible Superpoint features from a collection tangent image. That is, only returns the keypoints visible to a spherical camera at the center of the icosahedron.

    tex_image: 3 x N x H x W
    corners: N x 4 x 3 coordinates of tangent image corners in 3D
    image_shape: (H, W) of equirectangular image that we render back to
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [visible_kp, visible_desc] (M x 4, M x length_descriptors)
    """

    # ----------------------------------------------
    # Compute descriptors for each patch
    # ----------------------------------------------
    kp_list = []  # Stores keypoint coords
    desc_list = []  # Stores keypoint descriptors
    quad_idx_list = []  # Stores quad index for each keypoint
    for i in range(tex_image.shape[1]):
        #print(i)
        img = tex_image[:, i, ...]

        if opt == 'superpoint':
            img = process_img(img)
            kp_details = computes_superpoint_keypoints(img, opt)

        if opt == 'sift':
            kp_details = computes_sift_keypoints(img)

        if opt == 'orb':
            kp_details = computes_orb_keypoints(img)

        if opt == 'surf':
            kp_details = computes_surf_keypoints(img)


        if kp_details is not None:
            valid_mask = get_valid_coordinates(base_order,
                                               sample_order,
                                               i,
                                               kp_details[0][:, :2],
                                               return_mask=True)[1]
            visible_kp = kp_details[0][valid_mask]
            visible_desc = kp_details[1][valid_mask]

            # Convert tangent image coordinates to equirectangular
            visible_kp[:, :2] = convert_spherical_to_image(
                torch.stack(
                    convert_tangent_image_coordinates_to_spherical(
                        base_order, sample_order, i, visible_kp[:, :2]), -1),
                image_shape)

            kp_list.append(visible_kp)
            desc_list.append(visible_desc)

    all_visible_kp = torch.cat(kp_list, 0).float()  # M x 4 (x, y, s, o)
    all_visible_desc = torch.cat(desc_list, 0).float()  # M x 128

    # If top top and bottom of image is padding
    crop_h = compute_crop(image_shape, crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (all_visible_kp[:, 1] > crop_h) & (all_visible_kp[:, 1] <
                                              image_shape[0] - crop_h)
    all_visible_kp = all_visible_kp[mask]  # M x 4
    all_visible_desc = all_visible_desc[mask]  # M x 128
    return all_visible_kp, all_visible_desc


def keypoint_equirectangular(img, opt ='superpoint', crop_degree=0):
    """
    img: torch style (C x H x W) torch tensor
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [erp_kp, erp_desc] (M x 4, M x number_descriptors)
    """

    # ----------------------------------------------
    # Compute descriptors on equirect image
    # ----------------------------------------------

    if opt == 'superpoint':
        img = process_img(img)
        erp_kp_details = computes_superpoint_keypoints(img, opt)

    if opt == 'sift':
        erp_kp_details = computes_sift_keypoints(img)

    if opt == 'orb':
        erp_kp_details = computes_orb_keypoints(img)

    if opt == 'surf':
        erp_kp_details = computes_surf_keypoints(img)


    erp_kp = erp_kp_details[0]
    erp_desc = erp_kp_details[1]

    # If top top and bottom of image is padding
    crop_h = compute_crop(img.shape[-2:], crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (erp_kp[:, 1] > crop_h) & (erp_kp[:, 1] < img.shape[1] - crop_h)
    erp_kp = erp_kp[mask]
    erp_desc = erp_desc[mask]

    return erp_kp, erp_desc

def nn_match_two_way(desc1, desc2, nn_thresh = 0.7):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    desc1 /= np.linalg.norm(desc1, axis=0)[np.newaxis, :]
    desc2 /= np.linalg.norm(desc2, axis=0)[np.newaxis, :]

    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    #print(dmat)
    #print(np.max(dmat))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    #print(scores)
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    #return np.sort(matches, axis=1)
    return matches

def process_image_to_keypoints(image_path, corners, scale_factor, base_order, sample_order, opt, mode):

    img = load_torch_img(image_path)[:3, ...].float() # inputs/I1.png
    img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)

    # Resample the image to N tangent images (out: 3 x N x H x W)
    # N = 80, H, W depend of the sample_order( 2^(sample_order-1) = 512 )
    tex_image = create_tangent_images(img, base_order, sample_order).byte()

    if mode == 'tangent':
        tangent_image_kp, tangent_image_desc = keypoint_tangent_images(tex_image, base_order, sample_order, img.shape[-2:], opt , 0)

    if mode == 'erp':
        tangent_image_kp, tangent_image_desc = keypoint_equirectangular(img, opt)


    return tangent_image_kp, np.transpose(tangent_image_desc,[1,0])


