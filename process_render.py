import numpy as np
from numpy import *
import os
import argparse
from scipy import ndimage
from pdb import set_trace as pause
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

def synthesizeRotation(image, R):
    spi = SphericalImage(image)
    spi.rotate(R)
    return spi.getEquirectangular()

class SphericalImage(object):

    def __init__(self, equImage):
        self._colors = equImage.copy()
        self._dim = equImage.shape # height and width


        #image 2D to spherical coordinates
        phi, theta = meshgrid(linspace(0, pi, num = self._dim[0], endpoint=False), linspace(0, 2 * pi, num = self._dim[1], endpoint=False))
        print(phi.shape)
        #spherical coordinates to x,y,z coordinates
        self._coordSph = stack([(sin(phi) * cos(theta)).T,(sin(phi) * sin(theta)).T,cos(phi).T], axis=2)

    def rotate(self, R):
        # perform the rotation
        data = array(dot(self._coordSph.reshape((self._dim[0]*self._dim[1], 3)),R))
        self._coordSph = data.reshape((self._dim[0], self._dim[1], 3))


        x, y, z = data[:,].T

        # x, y ,z coordinates to spherical coordinates
        phi = arccos(z)
        theta = arctan2(y,x)
        theta[theta < 0] += 2*pi

        theta = self._dim[1]/(2*pi) * theta
        phi = self._dim[0]/pi * phi


        #interpolation
        if len(self._dim) == 3:
            for c in range(self._dim[2]): self._colors[...,c] = ndimage.map_coordinates(self._colors[:,:,c], [phi, theta], order=1, prefilter=False, mode='reflect').reshape(self._dim[0],self._dim[1])
        else:
            self._colors = ndimage.map_coordinates(self._colors, [phi, theta], order=1, prefilter=False, mode='reflect').reshape(self._dim[0],self._dim[1])
    def getEquirectangular(self): return self._colors
    def getSphericalCoords(self): return self._coordSph



def main():

    parser = argparse.ArgumentParser(description='Essential Matrix with RANSAC')
    parser.add_argument('dataset')
    parser.add_argument('--mode', default = "mixed")
    parser.add_argument('--samples', default = 100, type = int)
    args   = parser.parse_args()

    if args.mode == 'mixed':

        base_dir        = '/home/artcs/Desktop/Keypoints'

        low_dataset     = args.dataset.lower()
        canonical_text  = os.path.join('render',low_dataset+'_canonical.txt')
        canonical_image = os.path.join('render',low_dataset+'_canonical.png')
        data_can = np.loadtxt(canonical_text,skiprows=1)
        t0              = data_can[0,:]
        R               = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
        destiny_dir     = os.path.join('data', args.dataset+'_T')
        dataset_dir     = os.path.join('render',   args.dataset)
        name_image      = low_dataset

        paths = os.listdir(os.path.join('render',args.dataset))
        paths = [x[:-4] for x in paths]
        paths = list(set(paths))

        for i, path in enumerate(paths):
            print(path)
            os.system('mkdir -p '+destiny_dir+'/'+str(i)+'/')
            file_name = os.path.join(dataset_dir,path+'.txt')
            img_name  = os.path.join(dataset_dir,path+'.png')

            data = np.loadtxt(file_name, skiprows = 1)
            t1 = data[0,:]
            t = (t1-t0)#/np.linalg.norm(t1-t0)
            temp = t[0]
            t[0] = t[1]
            t[1] = temp
            t[2] = -t[2]

            np.save(os.path.join(base_dir,destiny_dir,str(i),'T'),t)
            np.save(os.path.join(base_dir,destiny_dir,str(i),'R'),R)

            os.system('cp '+canonical_image+' '+os.path.join(destiny_dir,str(i)))
            os.system('mv '+os.path.join(destiny_dir,str(i),low_dataset +'_canonical.png')+' '+os.path.join(destiny_dir,str(i),'O.png'))
            os.system('cp '+img_name+' '+os.path.join(destiny_dir,str(i)))
            os.system('mv '+os.path.join(destiny_dir,str(i),path+'.png')+' '+os.path.join(destiny_dir,str(i),'R.png'))

        path         = './data/'+args.dataset+'_T'
        destiny_path = './data/'+args.dataset

        for i in range(args.samples):
            print(i)
            num = int(np.random.uniform(0,len(paths),1))
            C = imageio.imread(os.path.join(path,str(num),'O.png'))
            I = imageio.imread(os.path.join(path,str(num),'R.png'))
            Rx = np.load(os.path.join(path,str(num),'R.npy'))
            Tx = np.load(os.path.join(path,str(num),'T.npy'))
            r = np.random.uniform(0,360,3)
            R1 = Rot.from_euler('xyz', r, degrees=True)
            R1 = R1.as_matrix()
            Ir = synthesizeRotation(I.copy(),R1)
            os.system("mkdir -p "+destiny_path+'/'+str(i))
            imageio.imwrite(os.path.join(destiny_path,str(i),'O.png'), C)
            imageio.imwrite(os.path.join(destiny_path,str(i),'R.png'), Ir)
            Rx = R1
            Tx = Rx@Tx
            np.save(os.path.join(destiny_path,str(i),'R.npy'),Rx)
            np.save(os.path.join(destiny_path,str(i),'T.npy'),Tx)

    elif args.mode == 'rotation':
        path = '/render'
        destiny_path = './data/'+args.dataset+'_R'

        I = imageio.imread("render/"+str(args.dataset.lower())+"_canonical.png")

        for i in range(args.samples):
            r = np.random.uniform(0,360,3)
            R1 = Rot.from_euler('xyz', r, degrees=True)
            Rx = R1.as_matrix()
            Tx = np.zeros(3)
            Ir = synthesizeRotation(I.copy(),Rx)

            os.system('mkdir -p '+destiny_path+'/'+str(i))

            imageio.imwrite(os.path.join(destiny_path,str(i),'O.png'), I.copy())
            imageio.imwrite(os.path.join(destiny_path,str(i),'R.png'), Ir)


            np.save(os.path.join(destiny_path,str(i),'R.npy'),Rx)
            np.save(os.path.join(destiny_path,str(i),'T.npy'),Tx)

    elif args.mode == 'translation':

        low_dataset     = args.dataset.lower()
        canonical_text  = os.path.join('render',low_dataset+'_canonical.txt')
        canonical_image = os.path.join('render',low_dataset+'_canonical.png')
        data_can = np.loadtxt(canonical_text,skiprows=1)
        t0              = data_can[0,:]
        R               = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
        destiny_dir     = os.path.join('data', args.dataset+'_T')
        dataset_dir     = os.path.join('render',   args.dataset)
        name_image      = low_dataset

        paths = os.listdir(os.path.join('render',args.dataset))
        paths = [x[:-4] for x in paths]
        paths = list(set(paths))

        for i, path in enumerate(paths):
            print(path)
            os.system('mkdir -p '+destiny_dir+'/'+str(i)+'/')
            file_name = os.path.join(dataset_dir,path+'.txt')
            img_name  = os.path.join(dataset_dir,path+'.png')

            data = np.loadtxt(file_name, skiprows = 1)
            t1 = data[0,:]
            t = (t1-t0)#/np.linalg.norm(t1-t0)
            temp = t[0]
            t[0] = t[1]
            t[1] = temp
            t[2] = -t[2]

            np.save(os.path.join(base_dir,destiny_dir,str(i),'T'),t)
            np.save(os.path.join(base_dir,destiny_dir,str(i),'R'),R)

            os.system('cp '+canonical_image+' '+os.path.join(destiny_dir,str(i)))
            os.system('mv '+os.path.join(destiny_dir,str(i),low_dataset +'_canonical.png')+' '+os.path.join(destiny_dir,str(i),'O.png'))
            os.system('cp '+img_name+' '+os.path.join(destiny_dir,str(i)))
            os.system('mv '+os.path.join(destiny_dir,str(i),path+'.png')+' '+os.path.join(destiny_dir,str(i),'R.png'))





if __name__ == '__main__':
    main()
