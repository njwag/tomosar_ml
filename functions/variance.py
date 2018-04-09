import numpy as np

class variance:
    
    """ Returns image of variance in a neighborhood. 
    The mean value for the neighborhood is assigned to its central pixel.
    """
    
    def slidingWindow(self,img,winSize,stepSize=1):
        #defining sliding window
        self.edge = (winSize-1)/2
        
        for y in xrange(0, img.shape[0], stepSize):
            for x in xrange(0, img.shape[1], stepSize):
                yield (x+self.edge, y+self.edge, img[y:y + winSize, x:x + winSize])
    

    def varImage(self,img,winSize,stepSize=1):
        
        assert winSize %2 == 1, "Window Size has to be odd number!"
        
        (width,height) = (img.shape[0],img.shape[1])
        variance = np.zeros([width,height])
        
        for (x, y, window) in self.slidingWindow(img, winSize):
            if window.shape[0] != winSize or window.shape[1] != winSize:
                continue
            variance[y,x] = np.var(window)
                    
        #filling edges of feature-array
        variance[:self.edge,:] = variance[(self.edge):(self.edge*2),:]
        variance[-self.edge:,:] = variance[(-self.edge*2):(-self.edge),:]
        variance[:,:self.edge] = variance[:,(self.edge):(self.edge*2)]
        variance[:,-self.edge:] = variance[:,(-self.edge*2):(-self.edge)]
        
        return(variance)