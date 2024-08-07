from pathlib import Path
import pandas as pd, numpy as np
import xarray as xr
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
# from skorch import NeuralNetClassifier
from skorch import NeuralNetRegressor

import torch.optim
from torch_optimizer import Lookahead


nm = 'current'

pth = Path.cwd()/'data'
assert pth.exists(), f'{pth} doesnt exist. Make sure your cwd is fugazey root directory'
# pth.mkdir(parents=True,exist_ok=True)
pth

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

camname = "c922 Pro Stream Webcam"
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

dscale = 1
camx,camy = 1920/dscale,1080/dscale
# we capture the first frame for the camera to adjust itself to the exposure
ret_val , cap_for_exposure = webcam.read()
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camx )
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)
webcam.set(cv2.CAP_PROP_EXPOSURE, 0.25)
webcam.set(cv2.CAP_PROP_EXPOSURE, 0.01)

xcenter=int(camx/2)
ycenter=660
yoffset = 325
xoffset = 450

resx,resy = pyautogui.size()
screen_w, screen_h = resx,resy

# pth = Path.cwd().parent/'data'
# pth.mkdir(parents=True,exist_ok=True)

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

    # frame = frame[ycenter-yoffset:ycenter+yoffset,xcenter-xoffset:xcenter+xoffset]
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
                font                   = cv2.FONT_HERSHEY_TRIPLEX
                bottomLeftCornerOfText = (10,40)
                fontScale              = 1
                fontColor              = (255,255,255)
                thickness              = 1
                lineType               = 2
                frame = cv2.putText(frame, plottext,
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType
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
    '''
    returns fullxy = xr.merge([normx,fully])\n
    normalized to xyz0, fullxy['head'] contains all 478 landmarks\n
    fullxy['mouse'] in screen coords (TODO 0-1)\n
    margin: px on edge of screen to leave out'''
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

    # yrng = np.linspace(0+margin,resx-margin,npts)
    # for x in np.linspace(0+margin,resx-margin,npts):
    #     moveTo(x,yrng[0],dt*3)
    #     for y in yrng:
    #         moveTo(x,y,dt,_pause=False)
    #         data += [ (snap() , np.array([x,y])) ]
    #     yrng = yrng[::-1]

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
    try:
        fullxy.to_netcdf(pth/'fullxy.nc')
    except Exception as e:
        print(f'WARNING: {pth/"fullxy.nc"} is open! \n{e}')
    print('done!')
    return fullxy
    # np.save(pth/'trainx.npy', trainx)
    # np.save(pth/'trainy.npy', trainy)
    # np.save(pth/'testx.npy', testx)
    # np.save(pth/'testy.npy', testy)

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
    scaler_x = scaler

    # fit and transform trainx
    scaler_x.fit(trainx_2D)

    trainx_scaled_2D = scaler_x.transform(trainx_2D)
    if len(x.shape)==3:
        trainx_scaled = trainx_scaled_2D.reshape(trainx.shape)
    elif len(x.shape)==2: #2d already
        trainx_scaled = trainx_scaled_2D
    assert trainx_scaled.shape == trainx.shape

    return trainx_scaled,scaler_x

def soultrain(fullxy=None, nm='current',crop=False):
    '''

    '''
    np.any(np.isnan(np.array([np.nan,3])))
    if not fullxy:
        fullxy = xr.open_dataset(pth/'fullxy.nc')
    sampdim = 's'
    fullxy
    fullxy['head'].coords
    # remove mouth pts
    if crop:
        toppts = np.array([  1, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 79, 93, 94, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 137, 139, 141, 142, 143, 144, 145, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 166, 168, 173, 174, 188, 189, 190, 193, 195, 196, 197, 198, 209, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 289, 290, 293, 294, 295, 296, 297, 298, 299, 300, 301, 305, 309, 323, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 362, 363, 366, 368, 370, 371, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 392, 398, 399, 412, 413, 414, 417, 419, 420, 425, 429, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477])
        croppd = fullxy['head'].isel(mark=toppts)
        fullxy = xr.merge( [fullxy['mouse'] , croppd ])
    fullxy
    toppts = np.array([  1, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 79, 93, 94, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 137, 139, 141, 142, 143, 144, 145, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 166, 168, 173, 174, 188, 189, 190, 193, 195, 196, 197, 198, 209, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 289, 290, 293, 294, 295, 296, 297, 298, 299, 300, 301, 305, 309, 323, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 362, 363, 366, 368, 370, 371, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 392, 398, 399, 412, 413, 414, 417, 419, 420, 425, 429, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477])
    len(toppts)
    nmarks = len(fullxy.mark)
    nmarks
    kernel_height = 3 # triangulate 3 pts
    padding_height = int((kernel_height - 1)/2) # because pytorch?
    3*16
    model = nn.Sequential(
        # First convolutional layer: considering 3 points at a time
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(kernel_height,kernel_height), 
                                        stride=1, padding=(padding_height, padding_height)) , 
        nn.ReLU(),
        
        # ... you can include more layers according to your desired architecture ...

        # Flatten before applying fully connected layers
        nn.Flatten(),

        # Fully connected layer (output size needs to be calculated based on the height after all convolutions)
        # The number of features before the fully connected layer depends on whether the convolution and pooling
        # layers are reducing the size of the point dimension or not.

        # Modify the number of output features in the last fully connected layer to 2 for regression
        # Assuming the height remains the same across layers due to padding, and the width is 1
        nn.Linear(nmarks * 16, 2)
    )

    assert isinstance(model,nn.Module)

    # `skorch` typically expects data in the format `(batch_size, channel, height, width)`
    #  for images in 2D convolutional layers,
    # which is similar to PyTorch's expectation.
    #  However, if you're using `xarray` and converting directly to PyTorch tensors, you have to
    #  make sure that the channels dimension is the second one
    chandim = 'crd'
    widthdim = 'mark'
    fullxy['head'] = fullxy['head'].transpose(sampdim,chandim,widthdim)
    fullxy
    scalexy = fullxy.copy()
    xdim='head'
    ydim='mouse'

    scalexy[xdim].values,scaler = scale(fullxy[xdim])
    # scalexy[xdim]

    scalexy[xdim].min(),scalexy[xdim].max()
    fullxy[xdim].min(),fullxy[xdim].max()
    xrlen = lambda da: float(da.count()) # if len is loading everything, maybe try this instead?

    def partition(ds,testpct=0.2,sampdim='s'):
        '''returns train,test'''
        if testpct>1:
            testpct = testpct/100
        trainpct = 1-testpct
        assert trainpct>0, trainpct
        assert trainpct<1, trainpct
        cutoff = int(trainpct * len(ds[sampdim]))
        train,test = ds.isel({sampdim:slice(None,cutoff)}) , ds.isel({sampdim:slice(cutoff,None)})

        assert len(train[sampdim]) + len(test[sampdim]) == len(ds[sampdim]), f'{len(train[sampdim])} + {len(test[sampdim])} != {len(ds[sampdim])}'
        return train,test
    # scale or not
    # trainxy,testxy = partition(fullxy)
    trainxy,testxy = partition(scalexy,0.1)
    trainxy

    # from torch.utils.data import Dataset

    # class xrda(Dataset,xr.DataArray):
    #     """A PyTorch Dataset that wraps an xarray DataArray\n
    #     this SHOULD work to train with BBBIG DATA lazily
    #     """
    #     def __init__(self, dataarray, sampdim='s', dtype=torch.float32):
    #         self.da = dataarray

    #     def __len__(self):
    #         return len(self.da)

    #     def __getitem__(self, idx):
    #         # This assumes the target variable is contained in the DataArray.
    #         x = torch.tensor(self.da.isel({sampdim:idx}).values, dtype=dtype)
    #         return x



    # x = xrda(trainxy['head'])
    # y = xrda(trainxy['mouse'])
    # import torch.from_numpy as tns
    tns = lambda da: torch.from_numpy( da.values.astype(np.float32) )
    trainy = tns(trainxy['mouse'])
    trainx = torch.from_numpy( np.expand_dims(
        trainxy['head'].values.astype(np.float32),
        -1
    ))
    # x.shape
    pos0 = np.array([0,10])
    pos1 = np.array([1,20])
    np.mean([pos0,pos1],axis=0)


    # custom optimizer to encapsulate Adam
    def make_lookahead(parameters, optimizer_cls, k, alpha, **kwargs):
        optimizer = optimizer_cls(parameters, **kwargs)
        return Lookahead(optimizer=optimizer, k=k, alpha=alpha)

    adam = dict(
        optimizer=make_lookahead,
        optimizer__optimizer_cls=torch.optim.Adam,
        optimizer__weight_decay=1e-2,
        optimizer__k=5,
        optimizer__alpha=0.5,
        lr=1e-3
    )

    net = NeuralNetRegressor(
        model,
        max_epochs=500,
        # criterion=mse,
        criterion=torch.nn.MSELoss,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        **adam
    )


    net.fit(trainx,trainy)  # y=None because dataset already generates both features and target
    # validate
    testx= torch.from_numpy( np.expand_dims(
        testxy['head'].values.astype(np.float32),
        -1
    ))
    testy = testxy['mouse'].values
    predy = net.predict(testx)
    # predy
    np.max(predy)
    trainxy['mouse'].max()
    rmse = np.sqrt( mse(testy,predy))
    print('RMSE error (px):',rmse)
    # --
    
    mdlpth = Path.cwd()/'fugazey'/'models'
    mdlpth.mkdir(parents=True,exist_ok=True)
    mdlpth

    net.save_params(f_params=mdlpth/f'{nm}.safetensors')
    # net.save_params(f_params=mdlpth/f'{nm}.safetensors', f_use_safetensors=True)
    # import pickle
    # with open(mdlpth/f'{nm}pkl', 'wb') as f:
    #     pickle.dump(net, f)

    joblib.dump(scaler, mdlpth/f'{nm}.gz')
    print(f"Trained model and scaler saved to {mdlpth}/f'{nm}.safetensors' and  {mdlpth}/f'{nm}.gz'")

if __name__=='__main__':
    snap(plotface=True)
    input('Hit enter once youre satisfied with settings')
    fullxy = gather(
        # dt=0
    )
    soultrain(fullxy,nm='current')

    # soultrain()

webcam.release()
cv2.destroyAllWindows()