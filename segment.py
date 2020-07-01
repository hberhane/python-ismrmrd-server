import tensorflow as tf
from dense_unet_model import Unet
import scipy.io as io
import os
import numpy as np

def segment(image):
	k = np.squeeze(image[0])
	#mrStruct = io.loadmat('MK/mag_struct.mat')
	print(k.shape)
	k = k[...,0]
	input_5 = tf.convert_to_tensor(k, dtype=tf.float32)
	input_5 = tf.expand_dims(input_5, dim=0)
	input_5 = tf.expand_dims(input_5, dim=4)
	flag = tf.placeholder(tf.bool)

	input_3p = tf.placeholder(tf.float32, shape=[1, 160,96,None, 1])
	#input_3p = tf.placeholder(tf.float32, shape=[1, 160,130,None, 1])

	logits_3 = Unet(x = input_3p, training=flag).model

	#logits_2 = Unet2(x = input_2p, training=flag).model
	llogits_3 = tf.nn.softmax(logits_3)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
	with tf.Session(config=config) as sesss:
		sesss.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		#checkpoint1 = tf.train.get_checkpoint_state("E:/HB/scripts_and_stuff/aliasing/new_noise")
		#sesss.run(tf.local_variables_initializer())
		s = sesss.run([input_5])
		print(s[0].shape)
		saver.restore(sesss, tf.train.get_checkpoint_state("/opt/codes/python-ismrmrd-server/pre-trained").model_checkpoint_path)
		h , seg4 = sesss.run([logits_3, llogits_3], feed_dict={input_3p:s[0], flag: True})
	sesss.close()
	
	mask3 = np.squeeze(seg4[...,1])
	mask3 = mask3>0.5

	#mrStruct = mags['mrStruct']
	#mrStruct['mrStruct']['dataAy'][0,0] = mask3
	#mrStruct['mrStruct']['dim4'] = 'unused'
	#mrStruct['mrStruct']['dim5'] = 'unused'

	
	return mask3

