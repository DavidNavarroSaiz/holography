import cv2
import time
import math
import numpy as np
import os    
from scipy.fftpack import fft, fftfreq, fftshift,ifft,fft2,ifft2,ifftn,ifftshift
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

class HolographyTools():
    """
    A set of tools created to help people to apply holography digital image processing 
    All the credits to: Juan Pablo Piedrahita Quintero - Jorge García Sucerquia - Carlos Trujillo
    """ 
    def __init__(self):

        pass
    
    def point_src(self,M,z,x0,y0,lamda,dx):
        """
        Creates a point source that will become an spherical source of light

        Parameters:

            M: int

                the width of the object 
            z:
                Source to sample distance
            x0:
                Coordinates of the point source, usually (0,0)
            lambda: 
                Wavelenght
        return: 
            P :
                light source
        example:

            p=point_src(3,1.228e-3,0,0,532e-9,6.86e-3)
            
        """

        N = M        
        dy=dx
        m,n=np.meshgrid(np.linspace(1-M/2,M/2,M),np.linspace(1-N/2,N/2,N))
        k=2*np.pi/lamda
        r=np.sqrt(z**2+(m*dx-x0)**2+(n*dy-y0)**2)
        P=np.exp(1j*k*r)/r
        return P

    def holo_interpF(self,Min,lambda_,z,deltax):
        """
        Interpolation applied to the hologram 
        
        """

        #Neccesary parameters
        row,__=Min.shape
        
        #Coordinates transformation required
        Xc,Yr=np.meshgrid(np.linspace(0,row,row),np.linspace(0,row,row))
        X=np.dot(deltax,(Xc - row / 2)) / (np.dot(lambda_,z))
        Y=np.dot(deltax,(Yr - row / 2)) / (np.dot(lambda_,z))
        Xp=np.dot(np.dot(lambda_,X),z) / np.sqrt(1 - np.dot((lambda_ ** 2),(X ** 2 + Y ** 2)))
        Yp=np.dot(np.dot(lambda_,Y),z) / np.sqrt(1 - np.dot((lambda_ ** 2),(X ** 2 + Y ** 2)))
        
        #Length of the coordinates matrix in pixels
        limit=np.ceil(np.max(np.max(Xp / deltax)))
        
        #Displace the origin of coordinates
        Xp=Xp - np.min(np.min(Xp))
        Yp=Yp - np.min(np.min(Yp))
        #Coordinates in pixel units
        Xp_p=(Xp) / deltax 
        Yp_p=(Yp) / deltax 
        
        #Positions to asign the displacement values
        iXp=(np.floor(Xp_p)).astype('int64')
        iYp=(np.floor(Yp_p)).astype('int64')
        
        #Calculate the weights for the neares
        x1frac=(iXp + 1.0) - Xp_p    
        x2frac=1.0 - x1frac    
        y1frac=(iYp + 1.0) - Yp_p
        y2frac=1.0 - y1frac
        x1y1=(np.multiply(x1frac,y1frac))  
        x1y2=(np.multiply(x1frac,y2frac)) 
        x2y1=(np.multiply(x2frac,y1frac))
        x2y2=(np.multiply(x2frac,y2frac))
        
        #Pre-allocate the interpolated hologram
        
        Mout=np.zeros((int(np.dot(2,limit)),int(np.dot(2,limit))),dtype=complex)
        
        #Interpolation process
        for it in np.arange(0,row-1):
            
            for jt in np.arange(0,row-1):          
                
                Mout[iYp[it,jt],iXp[it,jt]]=Mout[iYp[it,jt],iXp[it,jt]] + x1y1[it,jt]*Min[it,jt]    
                Mout[iYp[it,jt],iXp[it,jt] + 1]=Mout[iYp[it,jt],iXp[it,jt] + 1] + x2y1[it,jt]*Min[it,jt]
                Mout[iYp[it,jt] + 1,iXp[it,jt]]=Mout[iYp[it,jt] + 1,iXp[it,jt]] + x1y2[it,jt]*Min[it,jt]
                Mout[iYp[it,jt] + 1,iXp[it,jt] + 1]=Mout[iYp[it,jt] + 1,iXp[it,jt] + 1] + x2y2[it,jt]*Min[it,jt]
        
        #Crop to the size of the camera    
        Mout=Mout[int((np.dot(2,limit) - row) / 2):int(((np.dot(2,limit) + row) / 2)),int((np.dot(2,limit) - row) / 2):int(((np.dot(2,limit) + row) / 2))]
        return Mout



    def filtcosenoF(self, par, fi): 
        """
        Normalize coordinates in the interval (-pi, pi) and creates filters in horizontal and vertical direction

        Example : FC = filtcosenoF(100, 5)

        """

        #Coordinates
        Xfc,Yfc=np.meshgrid(np.linspace(- fi / 2,fi / 2,fi),np.linspace(fi / 2,- fi / 2,fi))
  
        FC1=np.cos(np.dot(np.dot(Xfc,(np.pi / par)),(1 / np.max(np.max(Xfc))))) ** 2
        FC2=np.cos(np.dot(np.dot(Yfc,(np.pi / par)),(1 / np.max(np.max(Yfc))))) ** 2
        #Intersect both directions
        FC=np.multiply(np.multiply(np.multiply((FC1 > 0),(FC1)),(FC2 > 0)),(FC2))
        #Re-scale from 0 to 1
        FC=FC / np.max(np.max(FC))
        
        return FC


    def prepairholoF(self,CH_m,xop,yop,Xp,Yp):

        """
        USER FUNCTION TO PREPARE THE HOLOGRAM USING NEAREST NEIGHBOOR INTERPOLATION

    
        """
        size=np.shape(CH_m)
        row = size[0]
        #New coordinates measured in units of the -2*xop/(row) pixel size
        Xcoord = (Xp - xop)/(-2*xop/(row))
        Ycoord = (Yp - yop)/(-2*xop/(row))
        #Find lowest integer
        iXcoord= (np.floor(Xcoord)).astype('int')
        iYcoord= (np.floor(Ycoord)).astype('int')
        #Assure there isn't null pixel positions

        iXcoord[iXcoord == 0 ]=1
        iYcoord[iYcoord == 0 ]=1
        # Calculate the fractioning for interpolation
        x1frac=(iXcoord + 1.0) - Xcoord    
        x2frac=1.0 - x1frac   
        y1frac=(iYcoord + 1.0) - Ycoord
        y2frac=1.0 - y1frac
        x1y1=np.multiply(x1frac,y1frac)    
        x1y2=np.multiply(x1frac,y2frac)
        x2y1=np.multiply(x2frac,y1frac)
        x2y2=np.multiply(x2frac,y2frac)
        
        #Pre-allocate the prepared hologram
        CHp_m=np.zeros((row,row),dtype = complex)
        #Prepare hologram (preparation1 - every pixel remapping)
        for it in range(row-1):
            for jt in range(row-1):
                CHp_m[iYcoord[it,jt]-1,iXcoord[it,jt]-1]= CHp_m[iYcoord[it,jt]-1,iXcoord[it,jt]-1] + (x1y1[it,jt])*CH_m[it,jt]
                CHp_m[iYcoord[it,jt]-1,iXcoord[it,jt]]= CHp_m[iYcoord[it,jt]-1,iXcoord[it,jt]] + (x2y1[it,jt])*CH_m[it,jt]
                CHp_m[iYcoord[it,jt],iXcoord[it,jt]-1]= CHp_m[iYcoord[it,jt],iXcoord[it,jt]-1] + (x1y2[it,jt])*CH_m[it,jt]
                CHp_m[iYcoord[it,jt],iXcoord[it,jt]]= CHp_m[iYcoord[it,jt],iXcoord[it,jt]] + (x2y2[it,jt])*CH_m[it,jt]   
        return CHp_m       




    def bluestein3(self,field,z,lamda,dx_in,dx_out):

        """
        Method used for the propagation of light

        parameters:
            dx_in:
                pixel size at sample plane
            
            dx_out:
                Hologram plane pixel pitch
        example: 
            b=bluestein3(p,1.228e-3,532e-9,6.86e-3,6.86e-3)

        """


        M,N=field.shape
        m,n=np.meshgrid(np.linspace(1-M/2,M/2,M),np.linspace(1-N/2,N/2,N))

        padx=int(M/2)
        pady=int(N/2)    
        
        dX=dx_out/np.sqrt(1-(dx_out/z)**2*(m**2+n**2))
        dY=dx_out/np.sqrt(1-(dx_out/z)**2*(m**2+n**2))
        
        
        k=2*np.pi/lamda
        R=np.sqrt(z**2 + (m * dX)**2 + (n * dY)**2)
            
        out_phase=1/(1j*lamda*R)*np.exp(1j*k*R)*np.exp(-1j*0.5*k/R*((dx_in*dX*m**2)+(dx_in*dY*n**2)))
            
        f1=field*np.exp(1j*0.5*k/R*((dx_in*(dx_in-dX)*m**2)+(dx_in*(dx_in-dY)*n**2)))
        f2=np.exp(1j*0.5*k/R*((dx_in*dX*m**2)+(dx_in*dY*n**2)))
        
        f1=np.pad(f1, ((padx, padx), (pady, pady)), 'constant',constant_values=0)
        f2=np.pad(f2, ((padx, padx), (pady, pady)), 'constant',constant_values=0)
        
        F1=np.fft.fft2(f1)
        F2=np.fft.fft2(f2)
        
        out=np.fft.fftshift(np.fft.ifft2(F1*F2))
        out=out[padx:padx+M,pady:pady+N]
        out=out_phase*out

        return out
  

    def Reconstruct_kreuzer3F(self,CH_m,z,L,lambda_,deltax,deltaX,FC,str):
        """
        Reconstruct an hologram to the original image

        
        Parameters:

            M: CH_m

                Input hologram (hologram-reference hologram)
            z:
                Source to sample distance
            L:
                source-to-screen distance
            lambda_:
                wavelenght
            deltax:
                Pixel size for the prepared hologram
            deltaX: 
                pixel width for the reconstructed image

            FC: 
                Result of filtcosenoF function

            str:
                "save" to save image
        return: 
            K :
                Reconstructed image
        
        """


        #Square pixels:
        deltaY=deltaX
        #Matrix size
        size=np.shape(CH_m)
        row = size[0]
        #Parameters
        k=(2*np.pi )/ lambda_
        W= deltax*row
        #Matriz coordinates
        X,Y=np.meshgrid(np.arange(0,row),np.arange(0,row))
        #Hologram origin coordinates
        xo=- W / 2
        yo=- W / 2
        #Prepared Hologram, coordinates origin
        xop=xo*L / math.sqrt(L ** 2 + xo ** 2)
        yop=yo*L / math.sqrt(L ** 2 + yo ** 2)
        #Pixel size for the prepared hologram
        deltaxp=xop / (- row / 2)
        deltayp=deltaxp
        #Coordinates origin for the reconstruction plane
        Yo=np.dot(- deltaX,(row)) / 2
        Xo=np.dot(- deltaX,(row)) / 2
        Xp=np.dot(np.dot((deltax),(X - row / 2)),L) / np.sqrt(L ** 2 + np.dot((deltax ** 2),(X - row / 2) ** 2) + np.dot((deltax ** 2),(Y - row / 2) ** 2))
        Yp=np.dot(np.dot((deltax),(Y - row / 2)),L) / np.sqrt(L ** 2 + np.dot((deltax ** 2),(X - row / 2) ** 2) + np.dot((deltax ** 2),(Y - row / 2) ** 2))
        #Search for prepared hologram if needed
        current_folder = os.getcwd()    
        file= os.path.join(current_folder, str)
        #Preparation of the hologram when neccesary    
        if os.path.exists(file) == False:
            CHp_m=self.prepairholoF(CH_m,xop,yop,Xp,Yp)    
    #    if exist(file,'file') == 0:
    #        #Prepare holo
        # CHp_m=prepairholoF(CH_m,xop,yop,Xp,Yp)
    ## kreuzer3F.m:48
    #        #    save(str,'CHp_m');
    #    else:
    #        #load .mat file with the saved prepared hologram
    #        load(str)
        
        #Multiply prepared hologram with propagation phase    
        Rp=np.sqrt((L ** 2) - (np.dot(deltaxp,X) + xop) ** 2 - (np.dot(deltayp,Y) + yop) ** 2)
        r=np.sqrt(np.dot((deltaX ** 2),((X - row / 2) ** 2 + (Y - row / 2) ** 2)) + (z) ** 2)
        CHp_m= CHp_m*((L/Rp)**4)*np.exp(-0.5*1j*k*(r**2 - 2*z*L)*Rp/(L**2))
        
        #Padding constant value
        pad=int(row / 2)
        #Padding on the cosine rowlter
        FC=np.pad(FC,(pad,pad), 'constant',constant_values=0)
        #Convolution operation
    #First transform
        T1=np.multiply(CHp_m,np.exp(np.dot((np.dot(1j,k) / (np.dot(2,L))),(np.dot(np.dot(np.dot(2,Xo),X),deltaxp) + np.dot(np.dot(np.dot(2,Yo),Y),deltayp) + np.dot(np.dot((X) ** 2,deltaxp),deltaX) + np.dot(np.dot((Y) ** 2,deltayp),deltaY)))))
        T1=np.pad(T1,(pad,pad), 'constant',constant_values=0)
        T1=fftshift(fft2(fftshift(np.multiply(T1,FC))))
        #Second transform
        T2=np.exp(np.dot(np.dot(- 1j,(k / (np.dot(2,L)))),(np.dot(np.dot((X - row / 2) ** 2,deltaxp),deltaX) + np.dot(np.dot((Y - row / 2) ** 2,deltayp),deltaY))))
        T2=np.pad(T2,(pad,pad), 'constant',constant_values=0)
        T2=fftshift(fft2(fftshift(np.multiply(T2,FC))))
        #Third transform
        K=ifftshift(ifftn(ifftshift(np.multiply(T2,T1))))
        K=K[pad+1:pad+row,pad+1:pad+row]

        
        return K

    def dlhm_sim(self,object,z,L,lambda_,dx):

        """
        DLHM_SIM Function to simulate inline holograms.
        This function simulates DLHM holograms using the methodology presented
    #   in doi:10.1364/AO.50.001745, it receives the geometrical parameters as
    #   inputs. Equal dimensions along the x- and y-axis is assumed.
        

        parameters:
            object:  
                Object to use as a sample 
            z:      
                 Source to sample distance
            L:
                Source to camera distance
            lambda:
                  Wavelength
            dx:
                  Hologram plane pixel pitch

            return:

            hologram:
                Hologram simulated of the object
            reference:
                Background of the system
            contrast: Background subtracted with hologram

            AN: Numeric aperture

        """


        #object size
        size=np.shape(object)
        M= size[0]
        M= size[1]
        #numerical aperture
        AN=np.sin(math.atan2((0.5*dx*M),L))
        #pixel size at sample plane
        dxo=(dx*z) / L
        #reference at sample plane
        ref_smp=self.point_src(M,z,0,0,lambda_,dxo)
        #propagation
        holo_field=self.bluestein3(np.multiply(object,ref_smp),L - z,lambda_,dxo,dx)
        ref_field=self.bluestein3(ref_smp,L - z,lambda_,dxo,dx)
        #hologram, reference and contrast hologram
        holo=np.abs(holo_field) ** 2
        ref=np.abs(ref_field) ** 2
        c_holo=holo - ref
        #outputs
        hologram=self.holo_interpF(holo,lambda_,L - z,dx)
        reference=self.holo_interpF(ref,lambda_,L - z,dx)
        contrast=self.holo_interpF(c_holo,lambda_,L - z,dx)

        
        return hologram,reference,contrast,AN
        