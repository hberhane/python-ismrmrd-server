import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#import time
from skimage import morphology
import scipy.io as io

#flow = io.loadmat('new_flow.mat')
#flow = flow['new_flow']
#mask = io.loadmat('aorta_mask_struct.mat')
#mask = mask['seg']
#mag = io.loadmat('mag.mat')
#mag = mag['mag']
def mipTesting(mask,flow,mag):
    flow = np.power(np.mean(np.square(flow), axis=3), 0.5)
    #imaspeed = np.power(np.mean(np.square(flow), axis=3), 0.5)
    #mag = flow

    # print(flow.shape)
    # print(mask.shape)

    mask = (morphology.remove_small_objects(np.squeeze(mask).astype(bool), 1000, in_place=True)).astype(float)

    # Mask the velocity magnitude
    mask = np.expand_dims(mask,axis=3)
    masked = np.multiply(flow,mask)
    #masked = np.zeros(flow.shape)

    #for tt in range(int(flow.shape[3])):
    #    masked[:, :, :, tt] = np.multiply(flow[:, :, :, tt],np.squeeze(mask))
    #print(masked.shape)

    # Choose a maximum velocity for the colorbar
    maxvel = 1.5

    # Make the mip
    mip = np.zeros([masked.shape[0], masked.shape[1], masked.shape[3]])
    mipmag = np.zeros([masked.shape[0], masked.shape[1], masked.shape[3]])
    for tt in range(int(masked.shape[3])):
        mip[:, :, tt] = np.max(np.squeeze(masked[:,:,:,tt]), axis=2)
        mipmag[:, :, tt] = np.max(np.squeeze(mag[:,:,:,tt]),axis=2)
    #print(mip.shape)

    # Scale the images
    mip[(mip > 1.5)] = 1.5
    mip[(mip < 0)] = 0
    mip = 2048 + (mip / 1.5)*2047 
    mipmag = (mipmag - np.amin(mipmag))
    mipmag = mipmag / np.amax(mipmag)
    mipmag = 2047 * mipmag

    # Superimpose the images
    mipimage = mipmag
    mipimage[(mip > 2048)] = mip[(mip > 2048)]



    mipimage = mipimage.astype(np.int16)
    io.savemat('thing.mat',{'thing':mipimage})
    return mipimage
"""
#Uncomment to display
fig = plt.figure()
for tt in range(int(mipimage.shape[2])):
    s = plt.imshow(np.squeeze(mipimage[:,:,tt]))
    ax = plt.gca()
    ax.axis("off")
    plt.show()
"""
