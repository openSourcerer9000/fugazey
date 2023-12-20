from pathlib import Path
import pandas as pd, numpy as np
import xarray as xr
import torch
import torch.nn as nn
kernel_height = 3 # triangulate 3 pts
padding_height = int((kernel_height - 1)/2) # because pytorch?
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
    nn.Linear(478 * 16, 2)
)

assert isinstance(model,nn.Module)
from sklearn.metrics import mean_squared_error as mse
# from skorch import NeuralNetClassifier
from skorch import NeuralNetRegressor

import torch.optim
from torch_optimizer import Lookahead

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
    max_epochs=20,
    # criterion=mse,
    criterion=torch.nn.MSELoss,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    **adam
)


from sklearn.preprocessing import StandardScaler

def scale(x,scaler_x='create',scalerFunc=StandardScaler):
    '''
    return trainx_scaled,scaler_x\n
    scaler_x either created from x, or passed in as arg\n
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
    print('TODO dynamic reshape, assumes 3D!')
    assert len(x.shape)==3
    trainx_2D = trainx.reshape(-1, trainx.shape[-1])

    # initialize scalers
    scaler_x = scalerFunc()

    # fit and transform trainx
    scaler_x.fit(trainx_2D)

    trainx_scaled_2D = scaler_x.transform(trainx_2D)
    trainx_scaled = trainx_scaled_2D.reshape(trainx.shape)
    assert trainx_scaled.shape == trainx.shape

    return trainx_scaled,scaler_x
pth = Path.cwd()/'data'
pth.mkdir(parents=True,exist_ok=True)
pth
fullxy = xr.open_dataset(pth/'fullxy.nc')
sampdim = 's'
fullxy
# scalexy = fullxy.copy()
# xdim='head'
# ydim='mouse'

# scalexy[xdim].values,scaler = scale(fullxy[xdim])
# scalexy[xdim]
# scalexy[xdim].min(),scalexy[xdim].max()
# fullxy[xdim].min(),fullxy[xdim].max()
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
trainxy,testxy = partition(fullxy)
trainxy
import xarray as xr
import torch
from torch.utils.data import Dataset

class xrda(Dataset,xr.DataArray):
    """A PyTorch Dataset that wraps an xarray DataArray\n
    this SHOULD work to train with BBBIG DATA lazily
    """
    def __init__(self, dataarray, sampdim='s', dtype=torch.float32):
        self.da = dataarray

    def __len__(self):
        return len(self.da)

    def __getitem__(self, idx):
        # This assumes the target variable is contained in the DataArray.
        x = torch.tensor(self.da.isel({sampdim:idx}).values, dtype=dtype)
        return x



x = xrda(trainxy['head'])
y = xrda(trainxy['mouse'])
# --
net.fit(x,y)  # y=None because dataset already generates both features and target
# Extending libraries to work with custom data formats typically involves wrapping your data with an adapter that makes the data compatible with the library's expectations. In the case of `skorch`, which is a scikit-learn compatible neural net library that wraps PyTorch, its usual input is either numpy arrays or PyTorch tensors.

# Since a big part of skorch's functionality involves handling data transformations and batching, a custom dataset wrapper that transforms xarray `DataArray` objects to PyTorch tensors would be the way to go. This can be achieved by subclassing `torch.utils.data.Dataset`.

# Here is an example of how you might build such a wrapper:
# ```

# Please note that the above implementation is quite simplified and assumes that dropping the target variable and converting the rest of the DataArray to a tensor represents the feature set for each observation appropriately.

# The actual implementation might involve more complex preprocessing, depending on the format of your DataArray (for example, you might have more complex dimensions you need to handle).

# After wrapping your xarray DataArray with `XarrayDataset`, `skorch` will be able to process it since `XarrayDataset` yields PyTorch tensors that `skorch` knows how to handle. When fitting the `skorch` model, you don't need to specify `y` separately since the dataset object already provides both `X` and `y`.

# Keep in mind, using a custom dataset requires careful handling of dimensions and potential metadata. This simplicity assumes a direct conversion is possible, but depending on the domain, more sophisticated processing might be needed to retain the integrity of the data's structure during the conversion.
