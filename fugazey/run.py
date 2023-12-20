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
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
import pickle
import joblib
print('libs loaded')

modelname = 'scaled0'
nmarks = 478
# toppts = np.array([  1, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 79, 93, 94, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 137, 139, 141, 142, 143, 144, 145, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 166, 168, 173, 174, 188, 189, 190, 193, 195, 196, 197, 198, 209, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 289, 290, 293, 294, 295, 296, 297, 298, 299, 300, 301, 305, 309, 323, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 362, 363, 366, 368, 370, 371, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 392, 398, 399, 412, 413, 414, 417, 419, 420, 425, 429, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477])
# nmarks = len(toppts)

mdlpth = Path(r'C:\app\fugazey\fugazey\models')
# netpkl=mdlpth/'scaled-val272.pkl'
netparams = mdlpth/f'{modelname}.safetensors'
scaler_gz=mdlpth/f'{modelname}.gz'

kernel_height = 3 # triangulate 3 pts
padding_height = int((kernel_height - 1)/2) # because pytorch?
model = nn.Sequential(
    # First convolutional layer: considering 3 points at a time
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(kernel_height,kernel_height), 
                                    stride=1, padding=(padding_height, padding_height)) , 
    nn.ReLU(),
    
    nn.Flatten(),

    nn.Linear(nmarks * 16, 2)
)

net = NeuralNetRegressor(
    module=model
    # criterion=torch.nn.NLLLoss,
)

net.initialize() # This is important!
net.load_params(
    f_params=netparams, 
    # f_optimizer='opt.pkl', f_history='history.json'
    )
# with open(netpkl, 'rb') as f:
#     net = pickle.load(f)
if scaler_gz:
    scaler = joblib.load(scaler_gz)
print('models loaded')

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

# shaep = (nmarks,3) # 3D , 478 landmarks from mp
def snap(plotface=True):
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
            cv2.imshow('ur face', frame)
            cv2.waitKey(1)
    else:
        pts = -np.ones( shaep )

    pts = xr.DataArray(pts,dims=('mark','crd'))
    assert len(pts['crd'])==3
    pts['crd'] = ['x','y','z']
    return pts

def getxyz0(da):
    '''this happens BEFORE cropping out any pts'''
    xyz0 = xr.concat([
        da.mean(dim=['s','mark']).sel(crd=['x','y']) ,
        da.min(dim=['s','mark']).sel(crd=['z'])
    ],dim='crd')
    return xyz0

def calibxyz0(nframes=5):
    ptz = [ snap() for _ in range(nframes) ]
    hed = xr.concat(ptz,dim='s')
    xyz0 = getxyz0(hed)
    return xyz0


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
import joblib 

from sklearn.preprocessing import StandardScaler


def scale(x,scaler=StandardScaler()):
    '''
    return trainx_scaled,scaler_x\n
    scaler either created from x, or passed in as arg\n
    returns return trainx_scaled,scaler_x\ns
    StandardScaler is default but you can pass in other func\n
    # Normalize your outputs by quantile normalizing or z scoring. To be rigorous, compute this transformation on the training data,
    #  not on the entire dataset. For example, with quantile normalization, if an example is in the 60th percentile of the training set,
    #  it gets a value of 0.6. (You can also shift the quantile normalized values down by 0.5 so that the 0th percentile is -0.5 and
    #  the 100th percentile is +0.5).
    # https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network'''

    # we'll treat the samples dimension independently and apply scaling on the features dimension
    trainx = x if isinstance(x,np.ndarray) else x.values
    # reshape trainx and trainy to 2D - sample axis, feature axis
    print('TODO dynamic reshape, assumes 2 or 3D!')
    assert len(x.shape)>1
    if len(x.shape)==3: #3d
        trainx_2D = trainx.reshape(-1, trainx.shape[-1])
    elif len(x.shape)==2: #2d already
        trainx_2D = x

    # initialize scalers
    scaler_x,_ = scaler

    # fit and transform trainx
    scaler_x.fit(trainx_2D)

    trainx_scaled_2D = scaler_x.transform(trainx_2D)
    if len(x.shape)==3:
        trainx_scaled = trainx_scaled_2D.reshape(trainx.shape)
    elif len(x.shape)==2: #2d already
        trainx_scaled = trainx_scaled_2D
    assert trainx_scaled.shape == trainx.shape

    return trainx_scaled,scaler_x

def nodfactor(pts,xyz0):
    '''
    
    '''
    pts = pts - xyz0
    x = pts.transpose('crd','mark')
    # assert x.shape == shaep[::-1], x.shape
    xscaled = scaler.transform(x.values)
    xtns = torch.from_numpy( np.expand_dims(
        np.expand_dims(
        xscaled.astype(np.float32),
        -1
        ),0
    ))

    pred = net.predict(xtns)
    # print(pred)
    pos1 = np.array(pred[0])
    # print(pos1)
    return pos1

def interp(pos0,pos1,speed=0.9):
# def move(pos0,pos1,speed=0.9,dt=0.05):
    pos = pos0 + speed * (pos1 - pos0)
    # moveTo(*pos,dt)
    return pos.copy()

import random


def listEqual(alist):
    '''returns True if every item in alist iterable is equal'''
    return all(alist[i+1] == alist[0] for i in range(len(alist)-1))


# def gather(nlines = 12,npts = 25,dt=0.05,margin=5):
#     '''margin: px on edge of screen to leave out'''

#     data = []

#     xrng = np.linspace(0+margin,resx-margin,npts)
#     # start from bottom to avoid neck strain
#     for y in np.linspace(0+margin,resy-margin,nlines)[::-1]:
#         moveTo(xrng[0],y,1)
#         for x in xrng:
#             moveTo(x,y,dt,_pause=False)
#             data += [ (snap() , np.array([x,y])) ]
#         xrng = xrng[::-1]

#     yrng = np.linspace(0+margin,resx-margin,npts)
#     for x in np.linspace(0+margin,resx-margin,npts):
#         moveTo(x,yrng[0],dt*3)
#         for y in yrng:
#             moveTo(x,y,dt,_pause=False)
#             data += [ (snap() , np.array([x,y])) ]
#         yrng = yrng[::-1]

#     # print(data)

#     # shuffling train + test
#     random.shuffle(data)
#     # cutoff = int(len(data)*.8)
#     # train,test = data[:cutoff],data[cutoff:]

#     assert listEqual([d[0].shape for d in data]), [d[0].shape for d in data]

#     fullx = np.stack([d[0] for d in data])
#     fully = np.stack([d[1] for d in data])

#     # trainx = np.stack([d[0] for d in train])
#     # trainy = np.stack([d[1] for d in train])
    
#     # testx = np.stack([d[0] for d in test])
#     # testy = np.stack([d[1] for d in test])

#     print(f'BOUNCING TO {pth}')

#     np.save(pth/'fullx.npy', fullx)
#     np.save(pth/'fully.npy', fully)
#     dimz = ('s','mark','crd')

#     fullx = xr.DataArray(fullx,dims=dimz)
#     fullx['crd'] = ['x','y','z']
#     #  # transform to put the origin on the mean face pt position xy, and the closest depth
#     #  this array of 3 pts may be calibrated to new user/position
#     #  this is part of preprocessing before fed to model
#     #  # will need to be further scaled for keras but OK
#     xyz0 = xr.concat([
#         fullx.mean(dim=['s','mark']).sel(crd=['x','y']) ,
#         fullx.min(dim=['s','mark']).sel(crd=['z'])
#     ],dim='crd')
#     xyz0
#     normx = fullx - xyz0
#     normx
#     # only for training data:
#     assert all(np.isclose(
#         normx.mean(dim=['s','mark']).sel(crd=['x','y']).values,
#         np.array([0,0])
#     )), normx.mean(dim=['s','mark']).sel(crd=['x','y'])
#     assert np.isclose( normx.min(dim=['s','mark']).sel(crd=['z']).values[0] , 0 ), \
#         normx.min(dim=['s','mark']).sel(crd=['z'])

#     assert fully.shape[-1]==2, f'should represent mouse x,y. hsape: {fully.shape}'
#     fully = xr.DataArray(fully,dims = ('s','pos'))
#     fully
#     fully.name = 'mouse'
#     fully = fully.to_dataset()
#     fully
#     fullx.name = 'head'
#     fullx = fullx.to_dataset()
#     fullx
#     fullxy = xr.merge([fullx,fully])
#     fullxy
#     fullxy.to_netcdf(pth/'fullxy.nc')
#     print('done!')
#     return fullxy
#     # np.save(pth/'trainx.npy', trainx)
#     # np.save(pth/'trainy.npy', trainy)
#     # np.save(pth/'testx.npy', testx)
#     # np.save(pth/'testy.npy', testy)

if __name__=='__main__':
    xyz0 = calibxyz0(nframes=5)
    print('calibrated to your current position')

    pos0 = np.array([resx/2,resy/2])
    rpos0 = np.array([0,0]) # relative
    tpos0 = pos0.copy()
    pupilcache = []

    # settings -----------------------------------------------------------
    def on_trackbar_change(coarsebuffer):
        # Here you would update your sequence with the new coarsebuffer value
        # print("New coarse buffer value:", coarsebuffer)
        pass
        # Use coarsebuffer to alter the processing in the while loop below
    # Create a named window
    cv2.namedWindow('ur face')
    # Initial value for coarsebuffer
    coarsebuffer = 20
    dt = 0#.05
    cachelen = 1
    eyescale = 100
    # Create a trackbar. Arguments: trackbar name, window name, value range and callback function
    cv2.createTrackbar('coarse buffer', 'ur face', 15, 30, on_trackbar_change)
    cv2.createTrackbar('eye scale', 'ur face', 200, 2000, on_trackbar_change)
    cv2.createTrackbar('lag', 'ur face', 0, 20, on_trackbar_change)
    cv2.createTrackbar('cache len', 'ur face', 1, 30, on_trackbar_change)
    # Dummy image just to have something to show in the window
    # You would replace this part with your video capture or sequence
    # dummy_frame = np.zeros((400, 400, 3), np.uint8)
    while True:
        # Check for trackbar position update
        coarsebuffer = cv2.getTrackbarPos('coarse buffer', 'ur face')
        dt = cv2.getTrackbarPos('lag', 'ur face')/10
        cachelen = cv2.getTrackbarPos('cache len', 'ur face')
        eyescale = cv2.getTrackbarPos('eye scale', 'ur face')
        # settings -----------------------------------------------------------

        pts = snap()
        
        mouthopen = pts.isel({'mark':slice(13,15)}).sel({'crd':'y'}).diff(dim='mark').values[0]
        mouthopen = mouthopen/0.05 # 0-1
        # print(mouthopen)

        # mouth joystick
        # pupl = 13
        # L = 308
        # R = 78
        # lt,rt,pupil = [ pts.isel({'mark':idx}).sel(crd=['x','y']).values for idx in (L,R,pupl) ]
        # og = np.mean((lt,rt), axis=0)
        # w = rt[0] - lt[0]
        # h = w*resy/resx # your eye given same aspect ratio of screen
        # # posnorm between -1:1 for x and y if you could move your eye all the way to the edge
        # posnorm = pupil - og
        # posnorm[0] = posnorm[0]/(2*w)
        # posnorm[1] = posnorm[1]/(2*h)

        # fine - these indices are on the uncropped pts
        pupl = 473
        # pupil = pts.isel({'mark':pupl}).values
        # avg of pupil and 4 iris control pts for less jitter from grainy webcam??
        pupil = pts.isel({'mark':slice(473,478)}).mean(dim='mark').values
        # mean of ALL pts!? not with mouth
        # pupil = pts.mean(dim='mark').values
        # pupil ranges from 0-1, now -1 to 1
        pupil = 2*pupil[:2] - 1
        pupilcache += [pupil]
        if len(pupilcache)>cachelen:
            pupilcache = pupilcache[-cachelen:]
        # pos1 = [
        #     posnorm[0] * eyescale*resx/2 + resx/2,
        #     posnorm[1] * eyescale*resy/2 + resy/2
        # ]
        #avg the last cachelen trail of jitters to smooth it?
        rpos1 = np.mean(pupilcache,axis=0) * eyescale
        # rpos1 = pupil * eyescale #x,y
        # rpos1 = posnorm * eyescale # mouth 

        rdist = np.sqrt(np.sum((rpos1-rpos0)**2))
        if rdist < coarsebuffer*2:
            # reduce jitter
            # rpos1 = interp(rpos0,rpos1,0.5 * (1-mouthopen))
            rpos1 = interp(rpos0,rpos1,0.7)
            # otherwise, move straight there to avoid lag when going across the screen

        # coarse
        # if 'crop' in modelname:
        #     # remove mouth pts
        #     pts = pts.isel(mark=toppts)
        pos1 = nodfactor(pts,xyz0)
        # 90% of the way there
        dist = np.sqrt(np.sum((pos1-pos0)**2))
        if dist < coarsebuffer:
            pos1 = pos0.copy() # nothin
        else:
            # print('pos changed')
            pos1 = interp(pos0,pos1,0.95)
            # print('interp',pos1)

        # print(pos1)
        # print(rpos1)
        # turn off pos abs
        # pos1 = np.array([resx/2,resy/2])
        # tpos1 = pos1.copy()# just abs
        tpos1 = pos1 + rpos1
        if np.any(np.isnan(tpos1)):
            print(f'NAN??? {tpos1}')
            # assert False
        else:
            moveTo(*tpos1,dt)
        tpos0 = tpos1.copy()
        pos0  = pos1.copy()
        rpos0 = rpos1.copy()
        # pos0 = move(pos0,pos1,0.3)
        
        # Break the loop if 'ESC' is pressed
        if cv2.waitKey(1) == 27:
            break

webcam.release()
cv2.destroyAllWindows()
