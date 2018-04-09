import numpy as np
from scipy.fftpack import fft, ifft

class frequencies:
    
    def kLargest(self,M,klargest):
        
        # corresponds to frequency
        f = sorted(range(len(M)), key=lambda x: M[x])[-klargest:]
        f = f[::-1]
        # corresponds to amplitude
        A = [M[i] for i in f]
        
        np.asarray(f)
        np.asarray(A)
        
        f = np.add(f,1)
        
        return(A)
    
    def importantf(self,tmgr,k=5):
        
        shape = (tmgr.shape[0],tmgr.shape[1],k)
        f = np.zeros(shape)
        # windowing using Hanning method to make signal periodic
        hann = np.hanning(tmgr.shape[-1])
        ft = fft(hann*tmgr)
        ft = 2*np.abs(ft[:,:,1:(ft.shape[-1]/2)])/((ft.shape[-1]/2)-1)
        
        for yi in range(0,tmgr.shape[0]):
            for xi in range(0,tmgr.shape[1]):
                f[yi,xi] = self.kLargest(ft[yi,xi],k)
        
        return(f)