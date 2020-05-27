import tensorflow as tf
from dense_unet_model import Unet
import scipy.io as io
import os
import numpy as np

def noise(image, new_flow):
	k = np.squeeze(image[0])
	print(k.shape)
	#input_4 = tf.convert_to_tensor(k, dtype=tf.float32)
	#input_4 = input_[32:160,...]
	#input_4 = tf.image.resize_image_with_crop_or_pad(input_4,128,96)
	#input_4 = (input_4 - tf.reduce_min(input_4))/(tf.reduce_max(input_4) - tf.reduce_min(input_4))
	#input_4 = tf.expand_dims(input_4, dim=0)
	#input_4 = tf.expand_dims(input_4, dim=4)
	#with graph.as_default():
	input_4 = tf.convert_to_tensor(k, dtype=tf.float32)
	input_4 = tf.expand_dims(input_4, dim=0)
	#input_4 = tf.expand_dims(input_4, dim=4)
	flag = tf.placeholder(tf.bool)

	input_2p = tf.placeholder(tf.float32, shape=[1, 160,96,None, 2])
	#input_2p = tf.placeholder(tf.float32, shape=[1, 160,130,None, 2])

	logits_2 = Unet(x = input_2p, training=flag).model

	#logits_2 = Unet2(x = input_2p, training=flag).model
	llogits_2 = tf.nn.softmax(logits_2)
	#sesss = tf.Session(graph=graph)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.4
	with tf.Session(config=config) as sesss:
		sesss.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		#checkpoint1 = tf.train.get_checkpoint_state("E:/HB/scripts_and_stuff/aliasing/new_noise")
		#sesss.run(tf.local_variables_initializer())
		s = sesss.run([input_4])
		print(s[0].shape)
		saver.restore(sesss, tf.train.get_checkpoint_state("./new_noise").model_checkpoint_path)
		h , seg3 = sesss.run([logits_2, llogits_2], feed_dict={input_2p:s[0], flag: True})
	sesss.close()
	tf.keras.backend.clear_session()

	mask2 = np.squeeze(seg3[...,1])
	mask2 = mask2>0.5
	#mask2 = np.logical_not(mask2)
	mask2 = mask2.astype(np.float32)
	mask2 = np.expand_dims(mask2,axis=3)
	mask2 = np.expand_dims(mask2,axis=4)

	final_flow = np.multiply(new_flow, mask2)
	#mrStruct1 = flows['mrStruct']
	#mrStruct['dataAy'][0,0] = final_flow

	#mag = mags['mrStruct']['dataAy'][0,0]
	#mag = mag[h1:HH-h1,w1:WW-w1,...]
	#mrStruct = mags['mrStruct']
	#mrStruct['dataAy'][0,0] = mag

	print(final_flow.shape)
	#io.savemat('MK/vel_struct.mat',{'mrStruct':mrStruct})
	#io.savemat('MK/mag_struct.mat',{'mrStruct':mrStruct})
	#io.savemat('MK/noise_mask.mat',{'noise':mask2})
	#del mrStruct1
	return image, final_flow
