from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.misc import imread, imsave
from skimage.feature import *
from skimage.filters import gabor,gabor_kernel
from skimage import io
from sklearn.feature_extraction import image
from sklearn.preprocessing import normalize
from scipy.fftpack import fft, ifft
from sklearn import preprocessing
from sklearn.feature_selection import (SelectPercentile, f_classif, SelectKBest, 
                                       RFE, RFECV, VarianceThreshold, chi2,
                                       mutual_info_classif)
from scipy import signal

import pickle
import copy
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import numpy.ma as ma
import pandas as pa
import time
import math
import scipy
from numpy.polynomial import polynomial
import collections
import random

import functions.ftfrequencies
import functions.correlation
import functions.pca
import functions.texture
import functions.variance

#%% Path variables
# Define paths for data and corresponding labels
datapath = ''
labelpath = ''
# change the res path according to your own directory.
respath = ''

#%% Loading Data

cov = '' # define covariance filename if applicable
          
tomo = '' # tomogram filename

lab = '' # labels filename

with h5.File(datapath + cov, 'r') as f:
    cc = f['data'][()]
    
tmgr = np.load(datapath + tomo)

#%% Loading reference Data

labels = np.load(datapath + lab)


#%% Defining features

def simple_entropy(tmgr):
    # values close to zero means there's only one dominant value
    # values closer to 1 mean that the power is equally distributed
    tmgr_norm = []
    for i in range(tmgr.shape[0]):
        tmgr_norm.append(normalize(tmgr[i],norm='l1',axis=1))
        
    tmgr_norm = np.asarray(tmgr_norm)
    
    entropy = [-i*(np.log(i)/np.log(200)) for i in tmgr_norm]
    entropy = np.asarray(entropy)
    entropy = np.nan_to_num(entropy)

    H = np.sum(entropy,axis=2)    

    return(H)


def hist(tmgr,axis=2):
    """The values of each 200-element vector of each pixel of the input array are 
    distributed over 14 bins (not equally spaced). The first 7 bins include all 
    values up to the mean while the following 7 bins include all values after the 
    mean. Each number represents the value count that falls into this bin."""
    
    n = int(np.sqrt(np.size(tmgr,axis=axis)))
    
    bins_1 = np.linspace(0, tmgr.mean(), num=(n/2), endpoint=False)
    bins_2 = np.linspace(tmgr.mean(), tmgr.max(), num=(n/2)+1, endpoint=True)
    bins = np.concatenate((bins_1, bins_2), axis=0)
    
    h = np.zeros((tmgr.shape[0],tmgr.shape[1],len(bins)-1))
    for i in range(tmgr.shape[0]):
        for j in range(tmgr.shape[1]):
            h[i,j],e = np.histogram(tmgr[i,j],bins=bins)
    
    return(h)


def polfit(tmgr,coef=7):
    # Fitting polynomials to data using least-squares method
    x = range(0,200)
    tmgr_pol = np.zeros((tmgr.shape[0],tmgr.shape[1],coef+1))
    for yi in range(0,tmgr.shape[0]):
        for xi in range(0,tmgr.shape[1]):
            tmgr_pol[yi,xi] = polynomial.polyfit(x,tmgr[yi,xi],coef)
            
    return(tmgr_pol)

#GLCM Features on Intensity Image
def getGLCM(cc_meanVV,winSize):
    if not isinstance(winSize,collections.Iterable):
        winSize = np.asarray((winSize,))
        
    GLCM = functions.texture.glcm_texture()
    glcm_list = []
    for i in winSize:
        glcm_list.append(GLCM.features(cc_meanVV,i))
    return(glcm_list)

#Gabor Filter Bank on Intensity Image
# Set orientation spacing to 30 deg
theta = np.asarray([0,30,60,90,120])*np.pi/180
# Set frequency ratio to sqrt(2)
frequency = np.asarray([0.05,0.05*math.sqrt(2),(0.05*math.sqrt(2))*math.sqrt(2)])
# Out: array([ 0.05      ,  0.07071068,  0.1       ])

def getGabor(img,theta,frequency):  
    gabor_im = []
    
    if not isinstance(theta,collections.Iterable):
        theta = np.asarray((theta,))
        
    if not isinstance(frequency,collections.Iterable):
        frequency = np.asarray((frequency,))
        
    for t in theta:
        for f in frequency:
            filt_real,filt_imag = gabor(img,frequency=f, theta=t)
            filt_total = filt_real + 1j * filt_imag
            gabor_im.append(filt_total)
    
    gabor_im = np.asarray(gabor_im)
    gabor_im = np.moveaxis(gabor_im, 0, -1)
    return(gabor_im)

# standard local statistics
tmgr_sum = np.sum(tmgr, axis=2)
tmgr_mean = np.mean(tmgr, axis=2)
tmgr_std = np.std(tmgr, axis=2)
tmgr_max = np.max(tmgr, axis=2)
tmgr_min = np.min(tmgr, axis=2)
tmgr_range = np.max(tmgr, axis=2)-np.min(tmgr, axis=2)

# coefficient of variation
tmgr_cov = np.mean(tmgr,axis=2)**2/np.std(tmgr,axis=2)**2

# Pearson Kurtosis
tmgr_kurt = scipy.stats.kurtosis(tmgr, axis=2,fisher=False)

# clipping at 2*mean to account for large spread of data
def clip(tmgr):
    tmgr2 = copy.deepcopy(tmgr)
    tmgr2[tmgr2>2*tmgr2.mean()] = 2*tmgr2.mean()
    return(tmgr2)

tmgr_sum = clip(tmgr_sum)
tmgr_mean = clip(tmgr_mean)
tmgr_std = clip(tmgr_std)
tmgr_max = clip(tmgr_max)
tmgr_min = clip(tmgr_min)
tmgr_cov = clip(tmgr_cov)
tmgr_kurt = clip(tmgr_kurt)
tmgr_range = clip(tmgr_range)

# position of max/min value
tmgr_argmax = np.argmax(tmgr, axis=2)
tmgr_argmin = np.argmin(tmgr, axis=2)

# Pearson's second skewness coefficient (median skewness)
tmgr_skew = (3*(np.mean(tmgr,axis=2)-np.median(tmgr,axis=2)))/np.std(tmgr,axis=2)

tmgr_en = simple_entropy(tmgr)
tmgr_hist = hist(tmgr)
tmgr_pol = polfit(tmgr)

ft = functions.ftfrequencies.frequencies()
tmgr_amp = ft.importantf(tmgr,k=5)

var = functions.variance.variance()

# Pearson correalation coefficient in a neighborhood
cor = functions.correlation.correlation()
tmgr_cor3 = cor.corImage(tmgr,3)
tmgr_cor7 = cor.corImage(tmgr,7)
tmgr_cor15 = cor.corImage(tmgr,15)

cc_meanVV = np.real(np.diagonal(cc,0,2,3).mean(2))
cc_meanVV = np.log(cc_meanVV)
cc_meanVV = 255*((cc_meanVV-np.min(cc_meanVV))/(np.max(cc_meanVV)-np.min(cc_meanVV)))
cc_meanVV = cc_meanVV.astype(int)

# GLCM Features on Covariance Matrix
cc_glcm3x3 = getGLCM(cc_meanVV,3)
cc_glcm5x5 = getGLCM(cc_meanVV,5)
cc_glcm7x7 = getGLCM(cc_meanVV,7)
cc_glcm15x15 = getGLCM(cc_meanVV,15)

# Gabor filter on Covariance Matrix
cc_gabor = getGabor(cc_meanVV,theta,frequency)
cc_gabor_mag = np.abs(cc_gabor) # Magnitude response
cc_gabor_sum = np.sum(cc_gabor_mag,axis=2)
cc_gabor_var_9x9 = var.varImage(cc_gabor_mag,9)
cc_gabor_var_15x15 = var.varImage(cc_gabor_mag,15)
cc_gabor_var_21x21 = var.varImage(cc_gabor_mag,21)



#%% Feature Vector

features_vect = [tmgr_sum,
               tmgr_mean,
               tmgr_std,
               tmgr_max,
               tmgr_min,
               tmgr_cov,
               tmgr_kurt,
               tmgr_range,
               tmgr_argmax,
               tmgr_argmin,
               tmgr_skew,
               tmgr_en,
               tmgr_hist,
               tmgr_amp,
               tmgr_pol,
               tmgr_cor3,
               tmgr_cor7,
               tmgr_cor15,
               cc_glcm3x3,
               cc_glcm5x5,
               cc_glcm7x7,
               cc_glcm15x15,
               cc_gabor_mag,
               cc_gabor_sum,
               cc_gabor_var_9x9,
               cc_gabor_var_15x15,
               cc_gabor_var_21x21]

def scale(tmgr):
    
    tmgr_scale = []
    
    if tmgr.ndim == 2:
        tmgr_scale = preprocessing.scale(tmgr)
        
    else:
        for i in range(tmgr.shape[2]):
            tmgr_scale.append(preprocessing.scale(tmgr[:,:,i]))
        
        tmgr_scale = np.asarray(tmgr_scale)
        tmgr_scale = np.moveaxis(tmgr_scale, 0, -1)
        
    return(tmgr_scale)

features = np.dstack(features_vect)


#%% CLASSIFICATION

# setting up classifiers
nb_clf = GaussianNB() # Gaussian Naive Bayes
rf_clf = RandomForestClassifier(n_estimators=30,criterion='gini',max_features='sqrt',
                                max_depth=None,min_samples_split=2,min_samples_leaf=1,
                                bootstrap=True,oob_score=False,n_jobs=1)
knn_clf = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto')
svm_clf = LinearSVC(penalty='l2',loss='squared_hinge',dual=False,tol=1e-4,C=1.0,
                    multi_class='ovr',max_iter=1000)

classifiers = [nb_clf, rf_clf, knn_clf, svm_clf]
classes = ['city', 'field', 'forest', 'grassland', 'street']
clf_labels = ['nb_clf', 'rf_clf', 'knn_clf', 'svm_clf']

def randomSelect(Y,k):
    mask = np.zeros(Y.shape, dtype=bool)
    
    for i in np.unique(Y):
        ix = random.sample(list(np.where(Y==i))[0],k)
        mask[ix] = True

    return(mask)

# masking for 0 values (unclassified)
msk = (labels != 0)
labels_msk = labels[msk]

height = labels.shape[0]
width = labels.shape[1]

choice = np.zeros(labels.shape,dtype=bool)
choice[:(height/2),(width/2):] = True
choice[(height/2):,:(width/2)] = True

it = [choice,~choice]

t = []
predictions = []
accuracy = []
confMat = []
importances = []

for clf in classifiers:
    y_predict = []
    y_labels = []
    start = time.time()
    for i in it:
        (x_train,y_train) = (features[i][labels[i] != 0],labels[i][labels[i] != 0])
        (x_test,y_true) = (features[~i][labels[~i] != 0],labels[~i][labels[~i] != 0])
        
        clf.fit(x_train,y_train)
        y_predict = y_predict + list(clf.predict(x_test))
        y_labels = y_labels + list(y_true)
        if clf.__class__.__name__ == 'RandomForestClassifier':
            importances.append(clf.feature_importances_)
        
    end = time.time() - start
    
    t.append(end)
    predictions.append(y_predict)
    accuracy.append(accuracy_score(y_labels, y_predict)*100)
    confMat.append(confusion_matrix(y_labels, y_predict))
    
#Classes: 1:city, 2:field, 3:forest, 4:grassland, 5:street, (0:unclassified)
cases_or,counts_or =  np.unique(labels_msk, return_counts=True)
accuracy_classes = []
for clf in range(len(classifiers)):
    for i in range(0,5):
        accuracy_classes.append((clf_labels[clf],((float(np.diag(confMat[clf])[i])/counts_or[i])*100),classes[i]))

# reconstructing image array
choice_vect = np.ndarray.flatten(choice)
msk_vect = np.ndarray.flatten(msk)
msk_recon1 = np.where((choice_vect == False) & (msk_vect == True))
msk_recon2 = np.where((choice_vect == True) & (msk_vect == True))
images = []
for i in range(len(predictions)):
    resimg = np.zeros(choice_vect.shape)
    count = np.count_nonzero(labels[~choice])
    resimg[msk_recon1] = predictions[i][:count]
    resimg[msk_recon2] = predictions[i][count:]
    images.append(resimg.reshape(labels.shape))
    
#Saving Image
fn = 'all_features'
fn = 'Results/' +fn
def saveImg(array,label):
    
    (rows,cols) = array.shape
    
    img = np.zeros((rows,cols,3))
    
    unclassified = [255, 255, 255] #white
    city = [255, 0, 0] #red
    field = [195,133,60] #sienna
    forest = [0, 102, 0] #dark green
    grasslands = [198, 199, 0] #light green
    street = [32, 32, 32] #dark grey
    
    img[np.where(array==0)] = unclassified
    img[np.where(array==1)] = city
    img[np.where(array==2)] = field
    img[np.where(array==3)] = forest
    img[np.where(array==4)] = grasslands
    img[np.where(array==5)] = street
    
    img = img.astype('uint8')
    io.imsave(fn+'_'+label+'.png',img)
    
    return(img)

for i in range(len(images)):
    saveImg(images[i],clf_labels[i])
    
file = open(fn+'_results.txt','w')
file.write('Classifiers: %s\n\n' %clf_labels)
file.write('Processing time (s): %s\n\n' %t)
file.write('Overall accuracy (%%): %s\n\n' %accuracy)
file.write('Class accuracies (%%): \n%s\n\n' %accuracy_classes)
file.write('Confusion Matrices: \n%s\n\n' %confMat)

if not not importances:
    file.write('Feature Importances: \n%s\n\n' %importances)
    
file.close()

# adding legend
for i in range(len(images)):
    im = io.imread(fn+'_'+clf_labels[i]+'.png')
    sizes = np.shape(im)
    height = float(sizes[0])
    width = float(sizes[1])
         
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    legend = plt.legend([mpatches.Patch(ec='#000000',fc='#FFFFFF',lw=0.1),
                mpatches.Patch(edgecolor='#000000',facecolor='#FF0000',linewidth=0.1),
                mpatches.Patch(ec='#000000',fc='#C3853C',lw=0.1),
                mpatches.Patch(ec='#000000',fc='#006600',lw=0.1),
                mpatches.Patch(ec='#000000',fc='#C6C700',lw=0.1), 
                mpatches.Patch(ec='#000000',fc='#202020',lw=0.1)], 
               ['unclassified','city','field','forest','grasslands','street'],
                loc=8, prop={'size': 1.1}, ncol=6, frameon=False, columnspacing=0.5)#labelspacing=0.2)
    
    
    ax.imshow(im)
    plt.savefig(fn+'_'+clf_labels[i]+'.png', dpi = height) 
    plt.close()
