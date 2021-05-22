import os   
import cv2
import time
import math
import numpy as np
from skimage import color
from skimage import io
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from holography_tools import HolographyTools

tools = HolographyTools()
plt.figure(figsize=(15,15))
# filename='/content/holoMHDL512.bmp'
samplename='./resources/cell.jpg'

sample = (color.rgb2gray(io.imread(samplename)))

smallest = np.amin(sample)
biggest = np.amax(sample)
if biggest> 2 : 
  maximo = 255
else:
  maximo = 1
sample = maximo- sample
sample = (cv2.resize(sample, (128,128),interpolation = cv2.INTER_AREA))
# print(biggest)
# sample=(plt.imread(samplename)).astype(float)
# sample = sample[0]
#geometric features
shape=np.shape(sample)

fi = shape[0]
z=  5e-05
# z=2e-03

L=0.12
lambda_= 405e-9
dx=(0.12288) / fi

hologram, reference, contrast, AN = tools.dlhm_sim( sample,z,L,lambda_,dx )


holoContrast=hologram - reference

plt.subplot(121)
plt.title('Image')
plt.imshow(sample,cmap='gray')
plt.subplot(122)
plt.title('Hologram')
plt.imshow(np.abs(hologram),cmap='gray')

plt.show()