import numpy as np
from skimage.feature import greycomatrix, greycoprops

class glcm_texture:
    
    """ Returns image of texture features based on the
    grey level co-occurence matrix.
    """
    
    def slidingWindow(self,img,winSize,stepSize=1):
        #defining sliding window
        self.edge = (winSize-1)/2
        
        for y in xrange(0, img.shape[0], stepSize):
            for x in xrange(0, img.shape[1], stepSize):
                yield (x+self.edge, y+self.edge, img[y:y + winSize, x:x + winSize])
    
    def features(self,img,winSize,stepSize=1):
        #calculating glcm-features
        assert winSize %2 == 1, "Window Size has to be odd number!"
        
        (width,height) = img.shape
        
        contrast = np.zeros([width,height])
        dissimilarity = np.zeros([width,height])
        homogeneity = np.zeros([width,height])
        ASM = np.zeros([width,height])
        energy = np.zeros([width,height])
        correlation = np.zeros([width,height])
        
        for (x, y, window) in self.slidingWindow(img, winSize):
            if window.shape[0] != winSize or window.shape[1] != winSize:
                continue
            glcm = greycomatrix(window,[1],[0],levels=256,symmetric=True,normed=True)
            contrast[y,x] = greycoprops(glcm, 'contrast')
            dissimilarity[y,x] = greycoprops(glcm, 'dissimilarity')
            homogeneity[y,x] = greycoprops(glcm, 'homogeneity')
            ASM[y,x] = greycoprops(glcm, 'ASM')
            energy[y,x] = greycoprops(glcm, 'energy')
            correlation[y,x] = greycoprops(glcm, 'correlation')
            
        glcm_feat = np.stack((contrast,dissimilarity,homogeneity,ASM,energy,correlation), axis=2)
        
        
        #filling edges of feature-array
        glcm_feat[:self.edge,:] = glcm_feat[(self.edge):(self.edge*2),:]
        glcm_feat[-self.edge:,:] = glcm_feat[(-self.edge*2):(-self.edge),:]
        glcm_feat[:,:self.edge] = glcm_feat[:,(self.edge):(self.edge*2)]
        glcm_feat[:,-self.edge:] = glcm_feat[:,(-self.edge*2):(-self.edge)]
        
        return(glcm_feat)