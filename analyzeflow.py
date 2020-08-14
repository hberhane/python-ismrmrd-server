
import matplotlib.pyplot as plt
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64
import re
import scipy.io as io
from eddy import eddy
from noise import noise
from segment import segment
#import matlab.engine
from aliasing import alias
#from makeMIP import makeMIP
from mipTesting import mipTesting

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    imgGroup = []
    magGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Image):
                meta = ismrmrd.Meta.deserialize(item.attribute_string)
                slice = extract_minihead_long_param(base64.b64decode(meta['IceMiniHead']).decode('utf-8'), 'AnatomicalPartitionNo')
                item.slice = slice

                # Only process phase images
                if item.image_type is ismrmrd.IMTYPE_PHASE:
                    imgGroup.append(item)
                elif item.image_type is ismrmrd.IMTYPE_MAGNITUDE:
                    magGroup.append(item)
                else:
                    # Group these images into separate series (to be fixed in FIRE later)
                    if meta['ImageType'] == 'ORIGINAL\\PRIMARY\\T1\\NONE':
                        item.image_series_index = 0
                    elif meta['ImageType'] == 'ORIGINAL\\PRIMARY\\VELOCITY\\NONE'  and  meta['FlowDirDisplay'] == 'FLOW_DIR_A_TO_P':
                        item.image_series_index = 1
                    elif meta['ImageType'] == 'ORIGINAL\\PRIMARY\\VELOCITY\\NONE'  and  meta['FlowDirDisplay'] == 'FLOW_DIR_R_TO_L':
                        item.image_series_index = 2
                    elif meta['ImageType'] == 'ORIGINAL\\PRIMARY\\VELOCITY\\NONE'  and  meta['FlowDirDisplay'] == 'FLOW_DIR_TP_IN':
                        item.image_series_index = 3
                    elif meta['ImageType'] == 'DERIVED\\PRIMARY\\ANGIO\\ADDITION':
                        item.image_series_index = 4

                    connection.send_image(item)
                    continue

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data 
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(imgGroup) > 0 and len(magGroup)>0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, magGroup, config, metadata)
            logging.debug("Sending images to client")
            connection.send_image(image)

    finally:
        connection.send_close()

def process_image(images, mag, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Incoming image data of type %s", ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Display MetaAttributes for first image
    tmpMeta = ismrmrd.Meta.deserialize(images[0].attribute_string)
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(tmpMeta))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in tmpMeta:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(tmpMeta['IceMiniHead']).decode('utf-8'))

    slices = [img.slice for img in mag]
    phases = [img.phase for img in mag]
    slice = [img.slice for img in images]
    phase = [img.phase for img in images]

    # Process each group of venc directions separately
    unique_venc_dir = np.unique([ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] for img in images])

    # Start the phase images at series 10.  When interpreted by FIRE, images
    # with the same image_series_index are kept in the same series, but the
    # absolute series number isn't used and can be arbitrary
    last_series = 10
    imagesOut = []
    imagesOutMip = []

    #if item.image_type is ismrmrd.IMTYPE_PHASE:
    datamag = np.zeros((mag[0].data.shape[2], mag[0].data.shape[3], max(
        slices)+1, max(phases)+1), mag[0].data.dtype)
    head = [[None]*(max(phases)+1) for _ in range(max(slices)+1)]
    meta2 = [[None]*(max(phases)+1) for _ in range(max(slices)+1)]

    for imgm, sli, phs in zip(mag, slices, phases):
        #if ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] == venc_dir:
        datamag[:, :, sli, phs] = imgm.data

    for venc_dir in unique_venc_dir:
        # data array has dimensions [x y sli phs]
        # info lists has dimensions [sli phs]
        data = np.zeros((images[0].data.shape[2], images[0].data.shape[3], max(slice)+1, max(phase)+1), images[0].data.dtype)
        head = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]
        meta2 = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]

        for img, sli, phs in zip(images, slice, phase):
            if ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] == venc_dir:
                data[:,:,sli,phs] = img.data
                head[sli][phs]    = img.getHead()
                meta2[sli][phs]    = ismrmrd.Meta.deserialize(img.attribute_string)

        logging.debug("Phase data with venc encoding %s is size %s" % (venc_dir, data.shape,))
        np.save(debugFolder + "/" + "data_" + venc_dir + ".npy", data)

        # Mask out data with high mean temporal diff
        threshold = 250
        data_meandiff = np.mean(np.abs(np.diff(data,3)),3)
        data_masked = data
        #data_masked[(data_meandiff > threshold)] = 2048
        np.save(debugFolder + "/" + "data_" + venc_dir + ".npy", data_masked)

        # Normalize and convert to int16
        data_masked = (data_masked.astype(np.float64) - 2048)*32767/2048
        data_masked = np.around(data_masked).astype(np.int16)
        if venc_dir == unique_venc_dir[0]:
            gh = np.zeros([data_masked.shape[0], data_masked.shape[1], data_masked.shape[2], 3, data_masked.shape[3]])
            gh[..., 0, :] = data
        elif venc_dir == unique_venc_dir[1]:
            gh[..., 1, :] = data
        elif venc_dir == unique_venc_dir[2]:
            gh[..., 2, :] = data
            meta = ismrmrd.Meta.deserialize(img.attribute_string)
            venc = meta['FlowVelocity']
            venc = int(venc)
            phaseRange = 4096

            tmpMask1 = np.ones(gh.shape)
            #print(tmpMask1.shape)
            tmpMask1 = tmpMask1*venc/100
            tmpMask2 = tmpMask1

            tmpMask1 = tmpMask1/(phaseRange/2)
            flows = gh*tmpMask1
            flows = flows - tmpMask2

            #TRANSPOSING
            flows = np.transpose(flows, (1,0,2,3,4))
            flows = np.flipud(flows)
            datamag = np.transpose(datamag, (1, 0, 2, 3))
            datamag = np.flipud(datamag)
            print("saving")
            #io.savemat('mag.mat', {'mag': datamag})
            #io.savemat('flow.mat', {'flow': flows})
            k,kk = eddy(datamag,flows)
            l, new_flow = noise(k,kk)
            #new_flow = alias(new_flow, venc)
            mm = segment(l)

            #UNTRANSPOSE
            flows = np.flipud(flows)
            flows = np.transpose(flows, (1,0,2,3,4))
            datamag = np.flipud(datamag)
            datamag = np.transpose(datamag, (1, 0, 2, 3))
            mm = np.flipud(mm)
            mm = np.transpose(mm, (1,0,2))
            new_flow = np.flipud(new_flow)
            new_flow = np.transpose(new_flow, (1,0,2,3,4))
            
            hh = (flows.shape[0] - new_flow.shape[0])//2
            h = flows.shape[0]
            ww = (flows.shape[1] - new_flow.shape[1])//2
            w = flows.shape[1]
            flows[hh:h-hh,ww:w-ww,...] = new_flow

            m = np.zeros([flows.shape[0], flows.shape[1], flows.shape[2]])
            m[hh:h-hh,ww:w-ww,...] = mm
            #io.savemat('aorta_mask_struct.mat',{'seg':m})
            #io.savemat('new_flow.mat', {'new_flow': flows})
            m = np.expand_dims(m,axis=3)
            #m = np.expand_dims(m,axis=4)
            mask = m

            mip = mipTesting(m,flows,datamag)
            #io.savemat('mip2.mat',{'mip':mip})
                        
            

        # Re-slice back into 2D images and let's also slice the MIP images
        
        for sli in range(data_masked.shape[2]):
            for phs in range(data_masked.shape[3]):
                # Create new MRD instance for the processed image
                tmpImg = ismrmrd.Image.from_array(data_masked[...,sli,phs].transpose())

                # Set the header information
                tmpHead = head[sli][phs]
                tmpHead.data_type          = tmpImg.getHead().data_type
                tmpHead.image_index        = phs + sli*data_masked.shape[3]
                tmpHead.image_series_index = last_series
                tmpImg.setHead(tmpHead)

                # Set ISMRMRD Meta Attributes
                tmpMeta = meta2[sli][phs]
                tmpMeta['DataRole']               = 'Image'
                tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
                tmpMeta['WindowCenter']           = '16384'
                tmpMeta['WindowWidth']            = '32768'

                xml = tmpMeta.serialize()
                logging.debug("Image MetaAttributes: %s", xml)
                tmpImg.attribute_string = xml
                imagesOut.append(tmpImg)

        



        last_series += 1
        
        


    #export the MIP
    sli = 0
    
    for phs in range(mip.shape[2]):
        # Create new MRD instance for the processed image
        tmpImg = ismrmrd.Image.from_array(mip[...,phs].transpose())

        # Set the header information
        tmpHead = head[sli][phs]
        tmpHead.data_type          = tmpImg.getHead().data_type
        tmpHead.image_index        = phs 
        tmpHead.image_series_index = last_series
        tmpImg.setHead(tmpHead)

        # Set ISMRMRD Meta Attributes
        tmpMeta = meta2[sli][phs]
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = '16384'
        tmpMeta['WindowWidth']            = '32768'

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)
    
    #last_series += 1


    return imagesOut

def extract_minihead_long_param(miniHead, name):
    # Extract a long parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamLong."' + name + r'">{)\s*\d*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    elif res.group(0).isspace():
        return 0
    else:
        return int(res.group(0))
