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
filename='./resources/holoMHDL512.bmp'
# filename='/content/Paramecium_Holo.bmp'
holo=(plt.imread(filename)).astype(float)
shape=np.shape(holo)
fi = shape[0]
# filenameref='/content/refMHDL512.bmp'
filenameref='./resources/refMHDL512.bmp'
# filenameref='/content/Paramecium_Reference.bmp'
ref=(plt.imread(filenameref)).astype(float)
holoContrast=holo - ref

# plt.subplot(251)
# plt.title('Holo')
# plt.imshow(holo,cmap='gray')
# plt.subplot(252)
# plt.title('reference')
# plt.imshow(ref,cmap='gray')
plt.subplot(121)
plt.title('holoContrast')
plt.imshow(holoContrast,cmap='gray')
# cv2.imshow("initial hologram",holo)
# cv2.waitKey()
# cv2.destroyAllWindows()
#geometric features
z=76e-6
L=12e-2
lambda_= 405e-9
dx=(12.288e-2) / fi
# z=5e-03
# L=12e-03
# lambda_=650e-09
# dx=(0.006875) / fi
#pixel size at reconstruction plance
deltaX=(z*dx) / L
  
#cosenus filter creation
FC=tools.filtcosenoF(100,fi)
#reconstrcut

start = time.time()
K=tools.Reconstruct_kreuzer3F(holoContrast,z,L,lambda_,dx,deltaX,FC,'save')
end = time.time()
print("time :",end - start)

amplitude = np.abs(K)
intensity = np.square(np.abs(K))
phase = np.angle(K)

# plt.subplot(254)
# plt.title('Amplitude')
# plt.imshow(amplitude,cmap='gray')

plt.subplot(122)
plt.title('intensity')
plt.imshow(intensity,cmap='gray')

# plt.subplot(256)
# plt.title('Phase')
# plt.imshow(phase,cmap='gray')

plt.show()