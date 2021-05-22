import cv2
import time
import math
import numpy as np
import os    
import matplotlib.pyplot as plt
from holography_tools import HolographyTools

tools = HolographyTools()
plt.figure(figsize=(20,20))
# filename='/content/holoMHDL512.bmp'
filename='./resources/G4_fly.bmp'
# filename='/content/Paramecium_Holo.bmp'
holo=(plt.imread(filename)).astype(float)
shape=np.shape(holo)
fi = shape[0]
# filenameref='/content/refMHDL512.bmp'
filenameref='./resources/GREF4_fly.bmp'
# filenameref='/content/Paramecium_Reference.bmp'
ref=(plt.imread(filenameref)).astype(float)
holoContrast=holo - ref

plt.subplot(251)
plt.title('Holo')
plt.imshow(holo,cmap='gray')
plt.subplot(252)
plt.title('reference')
plt.imshow(ref,cmap='gray')
plt.subplot(253)
plt.title('holoContrast')
plt.imshow(holoContrast,cmap='gray')

#geometric features
z=2.6e-03
L=12e-03
lambda_=650e-09
dx=(0.006875) / fi
#pixel size at reconstruction plance
deltaX=(z*dx) / L
  
#cosenus filter creation
print(fi)
FC=tools.filtcosenoF(100,fi)
#reconstrcut

start = time.time()
K=tools.Reconstruct_kreuzer3F(holoContrast,z,L,lambda_,dx,deltaX,FC,'save')
end = time.time()
print("time :",end - start)

amplitude = np.abs(K)
intensity = np.square(np.abs(K))
phase = np.angle(K)

plt.subplot(254)
plt.title('Amplitude')
plt.imshow(amplitude,cmap='gray')

plt.subplot(255)
plt.title('intensity')
plt.imshow(intensity,cmap='gray')

plt.subplot(256)
plt.title('Phase')
plt.imshow(phase,cmap='gray')


start = time.time()
hologram, reference, contrast, AN = tools.dlhm_sim( K,z,L,lambda_,dx )
end = time.time()
print("time :",end - start)

fil = np.abs(hologram)

# save this image
MAXAmp = np.max(np.max(fil))
MINAmp = np.min(np.min(fil))
FileAmp = (fil- MINAmp)/(MAXAmp-MINAmp)
FileAmp = 255*FileAmp

plt.subplot(257)
plt.title('hologram')
plt.imshow(np.abs(hologram),cmap='gray')

plt.subplot(258)
plt.title('constrast')
plt.imshow(np.abs(contrast),cmap='gray')

plt.subplot(259)
plt.title('reference')
plt.imshow(np.abs(reference),cmap='gray')


plt.show()