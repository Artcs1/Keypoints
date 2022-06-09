import numpy as np
import imageio
import cv2


from imageio import imread, imsave
from matplotlib import pyplot as plt

JET = plt.cm.ScalarMappable(norm=None, cmap='jet')
HOT = plt.cm.ScalarMappable(norm=None, cmap='hot')

def save_color_depth(nm, depth, min_depth=0., max_depth=1., cmap=JET):
    depth = np.clip(depth, min_depth, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = cmap.to_rgba(depth, norm=False)[:, :, :3]
    imsave(nm, np.uint8(depth * 255.))


# Generate dummy random image with float values
depth = np.load('Room.npy')*35
ind = np.where(depth>255)
depth[ind] = 255

print(np.sort(np.unique(depth)))
depth/=255
#depth = np.random.randn(512,1024)#load('Realistic.npy')
min_depth = np.maximum(0.01, np.min(depth))
max_depth = np.minimum(1., np.max(depth))
save_color_depth('_gt.png', depth, min_depth, max_depth)

#arr *= 30
#ind = np.where(arr > 255)
#arr[ind] = 255
#print(np.median(arr))
#print(np.sort(np.unique(arr)))
#
## freeimage lib only supports float32 not float64 arrays
#arr = arr.astype("uint8")
#
#print(np.max(arr))
#
#cv2.imshow('image',arr)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
