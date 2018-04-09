import numpy as np


class ppca: 
    
    """ Principal component Analysis: returns the set of principal components
    (linearly uncorrelated variables) from set of input observations
    """
    
    def preprocess(self,dataset, axis=0):
        
        mean_vect = dataset.mean(axis)
        mean_normal = dataset-mean_vect
        self.__mean_normal = mean_normal.T
        
        __cov = np.cov(self.__mean_normal)  
    
        self.__U,s,V = np.linalg.svd(__cov)
        
        m = len(s)
        check = 1
        optk = 1
		# finding optimal number of principal components
        for i in range(1,m):
            check = 1-((sum(s[0:i]))/sum(s))
            if(check>=0.01):
                optk = optk+1
                
        self.optk = optk
    
    def pca(self,k = 0):
        
        if k == 0:
            k=self.optk
        
        U_reduce = self.__U[:,:k]
        
        z = np.dot(U_reduce.T,self.__mean_normal)
        
        return(z.T)
