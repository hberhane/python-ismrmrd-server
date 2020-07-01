from dense_unet_model import Unet
import tensorflow as tf

import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.io as io
import argparse
#from keras import backend as K
f = []
fv = []

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def alias(flow, venc):
  mask = flow
  #print(mask.shape)
  w = mask.shape[1]
  w1 = int((w-96)/2)

  for fe in range(int(flow.shape[3])):
    for slices in range(int(flow.shape[4])):
      temp = flow[...,fe,slices]
      temp = temp.astype(np.float32)
      if fe != 1:
        f.append(temp)
      elif fe == 1:
        fv.append(temp)

  s = list(range(len(f)))
  #print(s)
  #random.shuffle(s)
  test_filename = 'f.tfrecords'
  writer = tf.python_io.TFRecordWriter(test_filename)

  for i in s:
      image_raw =  f[i].tostring()
      #label_raw = labels[i].tostring()
      #mask_raw = masks[i].tostring()

      height = f[i].shape[0]
      width = f[i].shape[1]
      depth = f[i].shape[2]
      Phases = len(f)
      features = {'test/image': _bytes_feature(image_raw),
                #'test/label': _bytes_feature(label_raw),
                #'test/mask': _bytes_feature(mask_raw),
                'test/height': _int64_feature(height),
                'test/depth':_int64_feature(depth),
                'test/phases': _int64_feature(Phases),
                'test/width': _int64_feature(width)}
      examples = tf.train.Example(features = tf.train.Features(feature = features))
      writer.write(examples.SerializeToString())
  s = list(range(len(fv)))
  #print(s)
  #random.shuffle(s)
  test_filename = 'fv.tfrecords'
  writer = tf.python_io.TFRecordWriter(test_filename)

  for i in s:
      image_raw =  fv[i].tostring()
      #label_raw = labels[i].tostring()
      #mask_raw = masks[i].tostring()

      height = fv[i].shape[0]
      width = fv[i].shape[1]
      depth = fv[i].shape[2]
      Phases = len(fv)
      features = {'test/image': _bytes_feature(image_raw),
                #'test/label': _bytes_feature(label_raw),
                #'test/mask': _bytes_feature(mask_raw),
                'test/height': _int64_feature(height),
                'test/depth':_int64_feature(depth),
                'test/phases': _int64_feature(Phases),
                'test/width': _int64_feature(width)}
      examples = tf.train.Example(features = tf.train.Features(feature = features))
      writer.write(examples.SerializeToString())

  writer.close()

  #path = "alis_og_t"
  #print('./'+path+'.tfrecords')
  #print("C:/Users/haben/Documents/aliasing/"+path)

  temp1 = aliasing(path1="f", path2="alis_og_t", venc=venc)
  temp2 = aliasing(path1="fv", path2="alis_og_x",venc=venc)
  #print(len(temp1))
  temp1 = np.stack(temp1, axis=3)
  temp2 = np.stack(temp2, axis=3)
  #print(temp2.shape)
  p = int(temp2.shape[3])
  #print(p)

  g = temp1[..., p-1]
  temp1[..., 1:p] = temp1[..., 0:p-1]
  temp1[..., 0] = g

  g = temp2[..., p-1]
  temp2[..., 1:p] = temp2[..., 0:p-1]
  temp2[..., 0] = g

  g = temp1[..., 2*p-1]
  temp1[..., p+1:2*p] = temp1[..., p:2*p-1]
  temp1[..., p] = g

  mask[10:138, w1:w-w1, :, 0, 0:p] = temp1[..., 0:p]
  mask[10:138, w1:w-w1, :, 1, 0:p] = temp2[..., 0:p]
  mask[10:138, w1:w-w1, :, 2, 0:p] = temp1[..., p:2*p]
  return mask

def feed_data(path):
    data_path = './'+path+'.tfrecords'  # address to save the hdf5 file
    feature = {'test/image': tf.FixedLenFeature([], tf.string),
               #'test/label': tf.FixedLenFeature([], tf.string),
               'test/depth': tf.FixedLenFeature([], tf.int64),
               'test/height': tf.FixedLenFeature([], tf.int64),
               'test/width': tf.FixedLenFeature([], tf.int64),
               #'test/venc': tf.FixedLenFeature([], tf.int64),
               'test/phases': tf.FixedLenFeature([], tf.int64)}
    
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    height = tf.cast(features["test/height"], tf.int32)
    #venc = tf.cast(features["test/venc"], tf.int64)
    width = tf.cast(features["test/width"], tf.int32)
    depth = tf.cast(features["test/depth"], tf.int32)
    phases = tf.cast(features["test/phases"], tf.int32)


    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['test/image'], tf.float32)
    
    
    image = tf.reshape(image, [height, width, depth])
    
    image = image[10:138,:,:]
    image = tf.image.resize_image_with_crop_or_pad(image, 128, 96)
    image = tf.cast(image,tf.float32)

    image = tf.expand_dims(image,axis = 0)
    image2 = tf.expand_dims(image,axis = 4)
    image = (image2 - tf.reduce_min(image2))/(tf.reduce_max(image2) - tf.reduce_min(image2))

    q = tf.FIFOQueue(capacity=50, dtypes=[tf.float32])
    enqueue_op = q.enqueue_many([image])
    qr = tf.train.QueueRunner(q,[enqueue_op])

    return image, image2, phases
def aliasing(path1,path2, venc):
	dd = []
	
	input_1p = tf.placeholder(tf.float32, shape=[1, 128,96,None, 1])
	image_batch, images, phase = feed_data(path = path1)
	
	flag = tf.placeholder(tf.bool)
	#g = tf.Graph()
	logits_1 = Unet(x = input_1p, training=flag).model
	saver1 = tf.train.Saver()


	llogits_1 = tf.nn.softmax(logits_1)
	checkpoint1 = tf.train.get_checkpoint_state("./"+path2)

	
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess = sess)
		saver1.restore(sess, checkpoint1.model_checkpoint_path)

		phases = sess.run(phase)
		#print(phases)
		for i in range(int(phases)):
			print(i)
			image,alias = sess.run([image_batch,images])#, input_2, input_3])
			_ , h = sess.run([logits_1, llogits_1], feed_dict={input_1p: image, flag: True})
			h = np.squeeze(h)
			alias = np.squeeze(alias)
			new_alis = alias
			h = h[...,1]>0.2
			for i in range(alias.shape[2]):
				for x in range(alias.shape[0]):
					for y in range(alias.shape[1]):
						if h[x,y,i] == 1:
							value = alias[x,y,i]
							if np.abs(value) < venc/100:
								new = value - (np.sign(value) * venc*2/100)
								new_alis[x,y,i] = new
							else:
								new = value - (np.sign(value) * venc*2/100)
								new = new - (np.sign(value) * venc*3/100)
								new_alis[x,y,i] = new
						else:
							continue
			dd.append(new_alis)

	sess.close()
	tf.keras.backend.clear_session()
	tf.reset_default_graph()
	

	return(dd)
	






