import cv2
import mediapipe as mp
from pathlib import Path
import pandas as pd, numpy as np
import  xarray as xr
import pyautogui
from pyautogui import moveTo
pyautogui.FAILSAFE = False
from pathlib import Path
import numpy as np

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camx,camy = 1920,1080
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camx )
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

xcenter=int(camx/2)
ycenter=660
yoffset = 325
xoffset = 450

resx,resy = pyautogui.size()
screen_w, screen_h = resx,resy

pth = Path.cwd().parent/'data'
pth.mkdir(parents=True,exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

for tri in range(50):
    _, frame = webcam.read()
    if frame is not None:
        break
else:
    assert False, 'ERROR, could not init webcam! make sure the permissions are on in settings, and a webcam is active'

def oldsnapnoxr(plotface=True):
# while True:
    shaep = (478,3) # 3D , 478 landmarks from mp
    _, frame = webcam.read()

    frame = frame[ycenter-yoffset:ycenter+yoffset,xcenter-xoffset:xcenter+xoffset]
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # print('len',len(landmarks))
        pts = np.array([[landmark.x,landmark.y,landmark.z] for landmark in landmarks])
        assert pts.shape==shaep, pts.shape

        if plotface:
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
            cv2.imshow('ur face', frame)
            cv2.waitKey(1)
    else:
        pts = -np.ones( shaep )
    
    # pts = xr.DataArray(pts,dims=('mark','crd'))
    # assert len(pts['crd'])==3
    # pts['crd'] = ['x','y','z']
    return pts

def getxyz0(da):
    xyz0 = xr.concat([
        da.mean(dim=['s','mark']).sel(crd=['x','y']) ,
        da.min(dim=['s','mark']).sel(crd=['z'])
    ],dim='crd')
    return xyz0

def snap(plotface=True,plottext=None):
# while True:
    shaep = (478,3) # 3D , 478 landmarks from mp
    _, frame = webcam.read()

    frame = frame[ycenter-yoffset:ycenter+yoffset,xcenter-xoffset:xcenter+xoffset]
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # print('len',len(landmarks))
        pts = np.array([[landmark.x,landmark.y,landmark.z] for landmark in landmarks])
        assert pts.shape==shaep, pts.shape

        # eye pos ----
        # pupl = 473
        # L = 398
        # R = 263
        # lt,rt,pupil = [
        #     np.array([landmarks[idx].x,landmarks[idx].y]) for idx in (L,R,pupl) ]
        # og = np.mean((lt,rt), axis=0)
        # w = rt[0] - lt[0]
        # h = w*resy/resx # your eye given same aspect ratio of screen
        # # posnorm between -1:1 for x and y if you could move your eye all the way to the edge
        # posnorm = pupil - og
        # posnorm[0] = posnorm[0]/(2*w)
        # posnorm[1] = posnorm[1]/(2*h)
        # -----------

        if plotface:
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if plottext:
                frame = cv2.putText(frame, plottext,
                    #  org, font,  
                #    fontScale, color, thickness, cv2.LINE_AA
                   ) 
            cv2.imshow('ur face', frame)
            cv2.waitKey(1)
    else:
        pts = -np.ones( shaep )

    pts = xr.DataArray(pts,dims=('mark','crd'))
    assert len(pts['crd'])==3
    pts['crd'] = ['x','y','z']
    return pts

            # if id == 1:
            #     screen_x = screen_w * landmark.x
            #     screen_y = screen_h * landmark.y
            #     pyautogui.moveTo(screen_x, screen_y)
        # left = [landmarks[145], landmarks[159]]
        # for landmark in left:
        #     x = int(landmark.x * frame_w)
        #     y = int(landmark.y * frame_h)
        #     cv2.circle(frame, (x, y), 3, (0, 255, 255))
        # if (left[0].y - left[1].y) < 0.004:
        #     pyautogui.click()
        #     pyautogui.sleep(1)
    # cv2.imshow('Eye Controlled Mouse', frame)
    # cv2.waitKey(1)


import random


def listEqual(alist):
    '''returns True if every item in alist iterable is equal'''
    return all(alist[i+1] == alist[0] for i in range(len(alist)-1))


def gather(nlines = 12,npts = 25,dt=0.05,margin=5):
    '''margin: px on edge of screen to leave out'''
    caption = 'Follow the mouse with your gaze until complete'

    data = []

    xrng = np.linspace(0+margin,resx-margin,npts)
    # start from bottom to avoid neck strain
    for y in np.linspace(0+margin,resy-margin,nlines)[::-1]:
        moveTo(xrng[0],y,1)
        for x in xrng:
            moveTo(x,y,dt,_pause=False)
            data += [ (snap(True,caption) , np.array([x,y])) ]
        xrng = xrng[::-1]

    yrng = np.linspace(0+margin,resx-margin,npts)
    for x in np.linspace(0+margin,resx-margin,npts):
        moveTo(x,yrng[0],dt*3)
        for y in yrng:
            moveTo(x,y,dt,_pause=False)
            data += [ (snap() , np.array([x,y])) ]
        yrng = yrng[::-1]

    # print(data)

    # shuffling train + test
    random.shuffle(data)
    # cutoff = int(len(data)*.8)
    # train,test = data[:cutoff],data[cutoff:]

    assert listEqual([d[0].shape for d in data]), [d[0].shape for d in data]

    fullx = xr.concat([d[0] for d in data],dim='s')
    fully = np.stack([d[1] for d in data])

    # trainx = np.stack([d[0] for d in train])
    # trainy = np.stack([d[1] for d in train])
    
    # testx = np.stack([d[0] for d in test])
    # testy = np.stack([d[1] for d in test])

    print(f'BOUNCING TO {pth}')

    # np.save(pth/'fullx.npy', fullx)
    # np.save(pth/'fully.npy', fully)
    dimz = ('s','mark','crd')

    # fullx = xr.DataArray(fullx,dims=dimz)
    # fullx['crd'] = ['x','y','z']
    #  # transform to put the origin on the mean face pt position xy, and the closest depth
    #  this array of 3 pts may be calibrated to new user/position
    #  this is part of preprocessing before fed to model
    #  # will need to be further scaled for keras but OK
    xyz0 = getxyz0(fullx)
    normx = fullx - xyz0
    normx
    # only for training data:
    assert all(np.isclose(
        normx.mean(dim=['s','mark']).sel(crd=['x','y']).values,
        np.array([0,0])
    )), normx.mean(dim=['s','mark']).sel(crd=['x','y'])
    assert np.isclose( normx.min(dim=['s','mark']).sel(crd=['z']).values[0] , 0 ), \
        normx.min(dim=['s','mark']).sel(crd=['z'])

    assert fully.shape[-1]==2, f'should represent mouse x,y. hsape: {fully.shape}'
    fully = xr.DataArray(fully,dims = ('s','pos'))
    fully
    fully.name = 'mouse'
    fully = fully.to_dataset()
    fully
    normx.name = 'head'
    normx = normx.to_dataset()
    fullxy = xr.merge([normx,fully])
    fullxy
    fullxy.to_netcdf(pth/'fullxy.nc')
    print('done!')
    return fullxy
    # np.save(pth/'trainx.npy', trainx)
    # np.save(pth/'trainy.npy', trainy)
    # np.save(pth/'testx.npy', testx)
    # np.save(pth/'testy.npy', testy)

if __name__=='__main__':
    gather()

webcam.release()
cv2.destroyAllWindows()
