import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

def makeMIP(flow,mask):
    # flow is the magnitude of the flow data (X Y Z T)
    # mask is the mask of the ROI

    mask = (morphology.remove_small_objects(np.squeeze(mask).astype(bool), 1000, in_place=True)).astype(float)

    # Mask the velocity magnitude
    masked = np.zeros(flow.shape)
    for tt in range(int(flow.shape[3])):
        masked[:, :, :, tt] = np.multiply(flow[:, :, :, tt], np.squeeze(mask))
    # print(masked.shape)

    # Choose a maximum velocity for the colorbar
    maxvel = 1.5

    # Make the mip
    mip = np.zeros([masked.shape[0], masked.shape[1], masked.shape[3]])
    for tt in range(int(masked.shape[3])):
        mip[:, :, tt] = np.max(np.squeeze(masked[:, :, :, tt]), axis=2)
    # print(mip.shape)

    # Print each timepoint
    fig = plt.figure()
    for tt in range(int(mip.shape[2])):
        s = plt.imshow(mip[:, :, tt], cmap='jet', vmin=0, vmax=maxvel)
        cbar = plt.colorbar(s)
        cbar.set_label('Velocity (m/s)', rotation=90)
        fig.tight_layout(pad=0)
        ax = plt.gca()
        ax.axis("off")
        plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()], antialiased=False)
        # matplotlib.rcParams['text.antialiased']=False
        fig.canvas.draw()
        # plt.show();

        datatemp = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        if tt == 0:
            temp = datatemp.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            mipimage = np.zeros([temp.shape[0], temp.shape[1], temp.shape[2], mip.shape[2]])
            mipimage[:, :, :, tt] = temp
        else:
            mipimage[:, :, :, tt] = datatemp.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Clear the figure for the next timestamp
        fig.clf()

    mipimage = mipimage.astype(int)

    # Uncomment to display
    # fig = plt.figure()
    # for tt in range(int(mip.shape[2])):
    # s = plt.imshow(np.squeeze(mipimage[:,:,:,tt]))
    # ax = plt.gca()
    # ax.axis("off")
    # plt.show();
    return mipimage