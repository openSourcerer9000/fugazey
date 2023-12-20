
import cv2
from gaze_tracking import GazeTracking
import pyautogui
from pyautogui import moveTo
pyautogui.FAILSAFE = False
from pathlib import Path
import numpy as np

gaze = GazeTracking()
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camx,camy = 1920,1080
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camx )
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

xcenter=int(camx/2)
ycenter=660
yoffset = 325
xoffset = 450

resx,resy = pyautogui.size()

dt = 0

pth = Path.cwd()/'data'
pth.mkdir(parents=True,exist_ok=True)

for tri in range(50):
    _, frame = webcam.read()
    if frame is not None:
        break
else:
    print('warning, could not init webcam!')

def snap():
    _, frame = webcam.read()
    frame = frame[ycenter-yoffset:ycenter+yoffset,xcenter-xoffset:xcenter+xoffset]
    
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    pupl,pupr = [-1,-1],[-1,-1] # if not found
    for itri,tri in enumerate(range(20)):
        if gaze.pupil_left_coords():
            pupl = gaze.pupil_left_coords()
        if gaze.pupil_right_coords():
            pupr = gaze.pupil_right_coords()
        if pupl!=[-1,-1] and pupr!=[-1,-1]:
            break
        print(f'{itri+2}th try')
        _, frame = webcam.read()
        frame = frame[ycenter-yoffset:ycenter+yoffset,xcenter-xoffset:xcenter+xoffset]
        gaze.refresh(frame)

    gazefeats = np.array(
                [[-1,-1,-1],
                 [-1,-1,-1]]
    )
    if gaze.eye_left and gaze.eye_right:
        gazefeats = np.array([ 
            [gaze.eye_left.origin[0],gaze.eye_right.origin[0],gaze.eye_left.blinking] ,
            [gaze.eye_left.origin[1],gaze.eye_right.origin[1],gaze.eye_right.blinking] 
            ])
    
    pupl = np.array(pupl).reshape(1,2)
    pupr = np.array(pupr).reshape(1,2)
    
    # print('partz',type(gaze.landmarks.parts()))
    # print(help(gaze.landmarks.parts))
    # print('rect',type(gaze.landmarks.rect))
    pts = np.array(
        [ [pt.x,pt.y] for pt in list(gaze.landmarks.parts())+[gaze.landmarks.rect.center()] ]
    )
    wh = np.array([[gaze.landmarks.rect.width(),gaze.landmarks.rect.height()]])
    print(wh)
    # for a in [pupl, pupr, gazefeats.T, pts, wh]:
    #     print(a.shape)
    # print(gazefeats.T)
    feats = np.concatenate( [pupl, pupr, gazefeats.T, pts, wh], 0  )
    # print('feats shape',feats.shape)
        
    return feats

import random


def listEqual(alist):
    '''returns True if every item in alist iterable is equal'''
    return all(alist[i+1] == alist[0] for i in range(len(alist)-1))


def gather():
    nlines = 3
    npts = 3

    data = []

    xrng = np.linspace(0,resx,npts)
    for y in np.linspace(0,resy,nlines):
        moveTo(xrng[0],y,dt*3)
        for x in xrng:
            moveTo(x,y,dt)
            data += (snap() , np.array([x,y]))
        xrng = xrng[::-1]

    yrng = np.linspace(0,resx,npts)
    for x in np.linspace(0,resx,npts):
        moveTo(x,yrng[0],dt*3)
        for y in yrng:
            moveTo(x,y,dt)
            data += (snap() , np.array([x,y]))
        yrng = yrng[::-1]

    # print(data)

    # shuffling train + test
    # random.shuffle(data)
    cutoff = int(len(data)*.8)
    train,test = data[:cutoff],data[cutoff:]

    assert listEqual([d[0].shape for d in data]), [d[0].shape for d in data]

    trainx = np.stack([d[0] for d in train])
    trainy = np.stack([d[1] for d in train])
    
    testx = np.stack([d[0] for d in test])
    testy = np.stack([d[1] for d in test])

    trainx.save(pth/'trainx.npy')
    trainy.save(pth/'trainy.npy')
    testx.save(pth/'testx.npy')
    testy.save(pth/'testy.npy')

if __name__=='__main__':
    gather()

# webcam.release()
# cv2.destroyAllWindows()
