import numpy as np

class correlation:
    
    """ Returns image of Pearson correalation coefficient in a neighborhood. 
    The mean value for the neighborhood is assigned to its central pixel.
    """
    
    def slidingWindow(self,img,winSize,stepSize=1):
        #defining sliding window
        self.edge = (winSize-1)/2
        
        for y in xrange(0, img.shape[0], stepSize):
            for x in xrange(0, img.shape[1], stepSize):
                yield (x+self.edge, y+self.edge, img[y:y + winSize, x:x + winSize])
    
    
    def pearsCor(self,window,add):
        #calculating Pearson correlation coefficient
        win = window.reshape(-1, window.shape[-1])
        value = (len(win)/2)+add
        center = win[value]
        
        correlation = np.corrcoef(win,center)[:,-1]
        correlation = np.append(correlation[:value],correlation[value+1:-1])
        cor_coef = correlation.mean()
        
        return(cor_coef)
    
    def corImage(self,img,winSize,stepSize=1):
        
        assert winSize %2 == 1, "Window Size has to be odd number!"
        
        if (winSize*2) %2 == 1:
            add = 1
        else:
            add = 0
            
        (width,height) = (img.shape[0],img.shape[1])
        correlation = np.zeros([width,height])
        
        for (x, y, window) in self.slidingWindow(img, winSize):
            if window.shape[0] != winSize or window.shape[1] != winSize:
                continue
            correlation[y,x] = self.pearsCor(window,add)
                    
        #filling edges of feature-array
        correlation[:self.edge,:] = correlation[(self.edge):(self.edge*2),:]
        correlation[-self.edge:,:] = correlation[(-self.edge*2):(-self.edge),:]
        correlation[:,:self.edge] = correlation[:,(self.edge):(self.edge*2)]
        correlation[:,-self.edge:] = correlation[:,(-self.edge*2):(-self.edge)]
        
        return(correlation)