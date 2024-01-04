import numpy as np
from math import radians, cos, sin, asin, sqrt
from scipy.spatial.distance import cdist
import os, sys
import tqdm


def loadPoses(path):
    poseData = np.loadtxt(path,float,delimiter=',',skiprows=1,usecols=[0,2,3])
    return poseData[:,0], poseData[:,1:]

def readImgTS(path,numIm=None):
    imgTs = np.loadtxt(path,float,delimiter=' ',usecols=[0])
    if numIm is None:
        numIm = len(imgTs)
    return imgTs[:numIm]

def getClosestPoseTsIndsPerImgTs(poseTs,imgTs,memEff=False):
    if memEff:
        matchInds = np.array([np.argmin(abs(poseTs-ts)) for ts in imgTs])
    else:
        diffMat = cdist(poseTs.reshape([-1,1]),imgTs.reshape([-1,1]))
        matchInds = np.argmin(diffMat,axis=0)
    return matchInds

def getClosestPoseTsIndsPerImgTs_searchLocal(poseTs,imgTs,searchRadius):
    # global look up for the first img timestamp
    firstIdx = np.argmin(abs(poseTs - imgTs[0]))
    matchInds = [firstIdx]
    print(firstIdx,matchInds)
    for i1 in range(1,len(imgTs)):
        lb = max(0,matchInds[-1] - searchRadius)
        ub = min(len(poseTs)-1, matchInds[-1]+ searchRadius)
        nextIdx = lb + np.argmin(abs(poseTs[lb:ub] - imgTs[i1]))
        matchInds.append(nextIdx)
    return np.array(matchInds)

def getImgPoses(posPath,imPath=None,numIm=None,searchRad=50,ret_tsMatchInds=False,imgTS=None):
    poseTS1, poseLatLon1 = loadPoses(posPath)
    if imgTS is None:
        imgTS = readImgTS(imPath,numIm)/1e9
    poseTS1 /= 1e9

    print(poseTS1.shape, poseLatLon1.shape, imgTS.shape)

    closeInds = getClosestPoseTsIndsPerImgTs(poseTS1,imgTS)
#     closeInds = getClosestPoseTsIndsPerImgTs_searchLocal(poseTS1,imgTS,searchRad)
    #print(closeInds)
    imgPoses1 = poseLatLon1[closeInds,:]
    #print(imgPoses1)
    
    if ret_tsMatchInds:
        return [imgPoses1, closeInds]
    else:
        return [imgPoses1]




def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 0.88915547 # this last factor has been calculated using 1-stereo from Google Maps

def haversineArr(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = [np.array(list(map(radians,arr))) for arr in [lon1, lat1, lon2, lat2]]

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r #*0.88915547 # this last factor has been calculated using 1-stereo from Google Maps

def cdist_haversine(ar2d_1,ar2d_2):
    """
    assumes ar2d as nx2 with lat lon as 1st and 2nd column
    """
    dists = []
    for i1 in tqdm.tqdm(range(len(ar2d_2)), total = len(ar2d_2)):
        val = haversineArr(ar2d_1[:,1],ar2d_1[:,0],ar2d_2[i1,1].reshape([1,-1]),ar2d_2[i1,0].reshape([1,-1]))
        dists.append(val)
    dists = np.array(dists)
    return dists.transpose()


def generateGTPairs(posPath1,imgPath1,posPath2,imgPath2,numImages,retImgPoseOnly=False,retTsMatchInds=False):
    """
    Input args:
    posPath* refers to the path to Oxford csv file like "$2014-07-14-14-49-50/gps/gps.csv"
    imgPath* refers to the path to Oxford image timestamps file like "$2014-07-14-14-49-50/stereo.timestamps"
    numImages limits the total number of images to be considered starting from index 0.
    retImgPoseOnly can be set to True if only pose info is needed for both the image datasets
    
    Returns imgPoses1 and imgPoses2 if retImgPoseOnly is True, else return gt12.
    gt12 comprises three columns as defined below:
    col 1 has the index from data 1 that is closest to the row index of gt12 (=index of data2)
    col 2 has the corresponding distance in meters
    col 3 has the flag set to True if distance between the pair is less than 5 meters
    """
    imPoses1 = getImgPoses(posPath1,imgPath1,numIm=numImages,ret_tsMatchInds=retTsMatchInds)
    imPoses2 = getImgPoses(posPath2,imgPath2,numIm=numImages,ret_tsMatchInds=retTsMatchInds)

    breakpoint()

    if retImgPoseOnly:
        return imPoses1, imPoses2
    
    dists_12 = cdist_haversine(imPoses1[0],imPoses2[0])
    minDists = np.min(dists_12,axis=0) * 1000
    gt_12 = np.vstack([np.argmin(dists_12,axis=0),minDists,minDists<5]).transpose()
    np.savetxt("./gt.txt",gt_12)
    


    return gt_12


if __name__ == "__main__":

    #timestamps 
    imgPath2= '/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo.timestamps'
    imgPath1 = '/hdd1/madhu/data/robotcar/2014-12-09-13-21-02/stereo.timestamps'

    posPath2 = "/home/madhu/code/feature-slam/git_repos/dtd/datasets/robotcar/2014-12-16-18-44-24_gps.csv"
    posPath1 = "/home/madhu/code/feature-slam/git_repos/dtd/datasets/robotcar/2014-12-09-13-21-02_gps.csv"

    generateGTPairs(posPath1,imgPath1,posPath2,imgPath2, None)

#numImages = 6000