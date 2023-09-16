import numpy as np
import argparse
import tqdm
import os
from os.path import join, exists
import matplotlib.pyplot as plt

def argParser():
    parser = argparse.ArgumentParser(description="Subsample Oxford Traverse and Obtain Splits")
    parser.add_argument("-p","--dataPath",type=str,help=("Path where raw data is stored"))
    parser.add_argument("-t","--traverseId",type=str,help=("timestamp id of the traverse, e.g. 2015-10-29-12-18-17"))
    parser.add_argument("-c","--camera",type=str,default="stereo",help=("type of camera, e.g. stereo/left"))
    parser.add_argument("-s","--split",type=str,help=("Specify split: train, test, val (default test=val for v2.2)"))
    parser.add_argument("-d","--drawPlots",action='store_true',help=("Plot Splits"))
    args = parser.parse_args()
    return args    

def readImgTS(path,numIm=None):
    imgTs = np.loadtxt(path,float,delimiter=' ',usecols=[0])
    if numIm is None:
        numIm = len(imgTs)
    return imgTs[:numIm]

def getClosestPoseTsIndsPerImgTs(poseTs,imgTs,memEff=True):
    if memEff:
        matchInds = np.array([np.argmin(abs(poseTs-ts)) for ts in imgTs])
    else:
        from scipy.spatial.distance import cdist
        diffMat = cdist(poseTs.reshape([-1,1]),imgTs.reshape([-1,1]))
        matchInds = np.argmin(diffMat,axis=0)
    return matchInds

def getDistsFromPoses(p1):
    return np.insert(np.linalg.norm(p1[1:,:] - p1[:-1,:],axis=1),0,0)

def getSpeedNormalizedInds(posDat,fixedDist=4,verbose=True):
    # dists = getDistsFromPoses(posDat)
    # inds2Remove = dists>10
    # if verbose:
    #     print("Total traversal distance: ",np.round(dists.sum()/1e3,3), " km")
    #     print("Removed Indices:",inds2Remove.sum())
    #     print(np.argwhere(inds2Remove).flatten())

    # posDat = posDat[~inds2Remove]
    dists = getDistsFromPoses(posDat)
    print(dists)
    print('Running through the loop of ', len(dists), 'for', fixedDist)
    inds = []

    for i in tqdm.tqdm(range(len(dists)), total = len(dists)):
        
        if dists[i] > fixedDist:
            inds.append(i)
        else:
            for j in range(i+1, len(dists)):
                if dists[i] + dists[j] > fixedDist:
                    inds.append(j)
                    break
                elif j == len(dists) - 1:
                    inds.append(j)
                    break

    print(dists[inds])
    return inds


    
    totalDist = np.sum(dists)
    requiredRange = np.arange(0,totalDist,fixedDist,float)
    indsVNorm = np.round(np.interp(requiredRange,np.cumsum(dists),np.arange(dists.shape[0]))).astype(int)
    indsVNormUnique = np.unique(indsVNorm)
    if verbose:
        print("Total traversal distance post inds removal: ",np.round(dists.sum()/1e3,3), " km")
        print(len(indsVNorm), " remaining after speed normalization out of ", len(posDat) )
        print("Unique indices: ", indsVNormUnique.shape)
    return indsVNormUnique

def getPoses_oxford_tsBased(oxRawDataPath,travTS,sampleDist=2,verbose=True,samplingType='odom',cam='stereo/left'):
    if 'stereo' in cam:
        cam = 'stereo'
    mainPosPath = "./{}_{}_timestamp_poses.csv"
    mainPosPath = mainPosPath.format(travTS,cam)
    if exists(mainPosPath):
        imgData = np.loadtxt(mainPosPath, delimiter=',')
        imgPoses, imgTS = imgData[:,1:], imgData[:,:1]
    else:
        insPath = join(oxRawDataPath,"{}/gps/rtk.csv".format(travTS))
        insData = np.loadtxt(insPath,delimiter=',',skiprows=1,usecols=[0,4,5])
        insTS, insNE = insData[:,0], insData[:,1:]
        imgTSPath = join(oxRawDataPath,"{}/{}.timestamps".format(travTS,cam))
        #imgTSPath = join(oxRawDataPath,"{}.timestamps".format(travTS))
        imgTS = readImgTS(imgTSPath)
        print("Searching for nearest poses for given timestamps...")
        closeInds = getClosestPoseTsIndsPerImgTs(insTS,imgTS,memEff=True)
        imgPoses = insNE[closeInds,:]
        #np.savetxt(mainPosPath,np.column_stack([imgTS,imgPoses]),delimiter=',')


    return imgPoses, imgTS

    
def getSplitInds_pointBased(thisPoses, s, ver='v2.1'):

    valCord, valRad = np.array([5735158.209064,  619861.387519]), 150
    if 'v2.2' in ver:
        valCord, valRad = np.array([5735298.209064,  619751.387519]), 270
    testCord, testRad = np.array([5734776.164594,  619832.666353]), 260
    offsetNonTrain, offsetTrain = 10, 30

    if ver == 'v2.1':
        if s == 'val':
            splitInds = np.prod(abs(thisPoses - valCord) <= (valRad-offsetNonTrain),axis=1,dtype=bool)
            splitInds = np.argwhere(splitInds).flatten()
        elif s == 'test':
            splitInds = np.prod(abs(thisPoses - testCord) <= (testRad-offsetNonTrain),axis=1,dtype=bool)
            splitInds = np.argwhere(splitInds).flatten()
        elif s == 'train':
            valInds_ = np.prod(abs(thisPoses - valCord) <= (valRad+offsetTrain),axis=1,dtype=bool)
            testInds_ = np.prod(abs(thisPoses - testCord) <= (testRad+offsetTrain),axis=1,dtype=bool)
            splitInds = np.argwhere((~valInds_)*(~testInds_)).flatten()
        elif s=='val+test':
            valInds = np.prod(abs(thisPoses - valCord) <= (valRad-offsetNonTrain),axis=1,dtype=bool)
            testInds = np.prod(abs(thisPoses - testCord) <= (testRad+offsetTrain),axis=1,dtype=bool)
            splitInds = np.union1d(np.argwhere(valInds).flatten(), np.argwhere(testInds).flatten())
    elif 'v2.2' in ver:
        if s == 'val' or s == 'test':
            valInds = np.prod(abs(thisPoses - valCord) <= (valRad-offsetNonTrain),axis=1,dtype=bool)
            testInds = np.prod(abs(thisPoses - testCord) <= (testRad+offsetTrain),axis=1,dtype=bool)
            splitInds = np.union1d(np.argwhere(valInds).flatten(), np.argwhere(testInds).flatten())
        elif s == 'train':
            valInds_ = np.prod(abs(thisPoses - valCord) <= (valRad+offsetTrain),axis=1,dtype=bool)
            testInds_ = np.prod(abs(thisPoses - testCord) <= (testRad+offsetTrain),axis=1,dtype=bool)
            splitInds = np.argwhere((~valInds_)*(~testInds_)).flatten()
        elif s == 'valFull':
            splitInds = np.arange(thisPoses.shape[0])

    return splitInds

def plotSplits(poseData,dataset='oxford',verOx='v2.2'):
    ms = dict(zip(['train', 'val', 'test'],[8,4,1]))
    for s in ['train', 'val', 'test']:
        if dataset == 'oxford':
            splitInds = getSplitInds_pointBased(poseData,s,verOx)
        plt.plot(poseData[splitInds][:,0], poseData[splitInds][:,1],'.-',label=s,ms=ms[s])
    plt.savefig('dataset_splits.png')
    plt.legend()
    plt.title("Data Splits")
    plt.show()

def getOxSplits(dataPath,split=None,travTS1=None,plotSplitsFlag=False,ver='v2.1', sampleDist=-1, verbose=True, samplingType='odom', cam='stereo/left'):

    if sampleDist == -1:
        sampleDist = 2    
    pos1, ts1 = getPoses_oxford_tsBased(dataPath,travTS1,sampleDist,verbose,samplingType,cam)
    print("Subsampled Traverse Shapes:", pos1.shape)

    if plotSplitsFlag:
        plotSplits(pos1,verOx=ver)
        plt.plot(getDistsFromPoses(pos1))
        plt.title("Distance between consecutive frames")
        plt.ylabel("Distance (in meters)")
        plt.xlabel("Frame Index")
        plt.show()

    splitInds1 = None
    if split is not None:
        splitInds1 = getSplitInds_pointBased(pos1,split,ver)
        split_poses = pos1[splitInds1]
        split_timestamps = ts1[splitInds1]
        plt.plot(split_poses[:,0],split_poses[:,1])
        plt.savefig('test_split.png')

        return split_poses, split_timestamps

        inds = []
        for i in tqdm.tqdm(range(split_poses.shape[0]),total = len(split_poses)):
            for j in range(i, split_poses.shape[0]):
                dist = np.linalg.norm(split_poses[i] - split_poses[j])
                if dist > sampleDist:
                    inds.append([i,j])
                    break
        inds = np.array(inds)
        
        origin = split_timestamps[inds[:,0]]
        destination = split_timestamps[inds[:,1]]

        print(np.linalg.norm(split_poses[inds[:,0]] - split_poses[inds[:,1]], axis=1))


        timestamps = np.stack([origin, destination], axis=1)

        return split_poses, splitInds1, inds, timestamps
        
       

if __name__=="__main__":
    args = argParser()

    # poses is two-column
    # sampleInds is a vector carrying indices obtained after sampling
    # splitInds is a vector carrying indices (wrt sampled poses after using sampleInds) for a particular split
    poses, timestamps = getOxSplits(args.dataPath, args.split, args.traverseId, args.drawPlots, cam=args.camera, sampleDist=10)



    #saveName = "_".join([args.traverseId,args.split])
    #np.savez(saveName,poses=poses,sampleInds=sampleInds,splitInds=splitInds,timestamps=timestamps)
    #print("Saved info: ",saveName)