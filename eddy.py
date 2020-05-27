import numpy as np

import os
import tensorflow as tf

import scipy.io as io
from dense_unet_model import Unet
from lsqf2 import lsqf2
gh = []
l = []
labels = []
label = []
k = []
la = []
masks = []
images = []
f = []

def eddy(mag, flow):
	

	imaspeed = np.power(np.mean(np.square(flow),axis=3), 0.5)
	minV = np.amin(0.7*np.reshape(mag,(-1,1)))
	maxV = np.amax(0.7*np.reshape(mag,(-1,1)))
	tmp = mag > maxV
	not_tmp = np.logical_not(tmp)
	mag = np.multiply(mag, not_tmp) + np.multiply(tmp, maxV)
	tmp = mag<minV
	not_tmp = np.logical_not(tmp)

	mag = np.multiply(mag, not_tmp) + np.multiply(tmp, minV)
	mag = (mag - minV)/(maxV - minV)
	pcmra = np.multiply(imaspeed, mag)
	pcmra = np.squeeze(np.mean(np.square(pcmra), axis = 3))

	std_flow = np.std(imaspeed, axis=3)
	#del flow
	#del mag

	
	print(pcmra.shape)
	print(std_flow.shape)
	pcmra = np.expand_dims(pcmra,axis=3)
	std_flow = np.expand_dims(std_flow, axis=3)
	X = np.concatenate((pcmra, std_flow), axis=3)
	X = X.astype(np.float32)
	del pcmra
	del std_flow

	
	for j in range(2):
		X[...,j] = (X[...,j] - np.amin(X[...,j]))/(np.amax(X[...,j]) - np.amin(X[...,j]))
	#X = X[10:138,...]	
	print(X.shape)
	

	input_ = tf.convert_to_tensor(X, dtype=tf.float32)
	input_ = tf.transpose(input_, perm=[2,0,1,3])

	input_ = tf.image.resize_image_with_crop_or_pad(input_, 160, 96)
	#input_ = tf.image.resize_image_with_crop_or_pad(input_, 160, 130)

	input_ = tf.transpose(input_, perm=[1,2,0,3])
	input_ = tf.squeeze(input_)
	#input_ = (input_ - tf.reduce_min(input_))/(tf.reduce_max(input_) - tf.reduce_min(input_))
	input_ = tf.expand_dims(input_, dim=0)
	#input_ = tf.expand_dims(input_, dim=4)
	input_p = tf.placeholder(tf.float32, shape=[None, 160,96,None, 2])
	#input_p = tf.placeholder(tf.float32, shape=[None, 160,130,None, 2])
	flag = tf.placeholder(tf.bool)
	logits_1 = Unet(x = input_p, training=flag).model
	saver1 = tf.train.Saver()
	llogits_1 = tf.nn.softmax(logits_1)

	checkpoint1 = tf.train.get_checkpoint_state("./new_eddy")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.4
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		image = sess.run([input_])
		saver1.restore(sess, checkpoint1.model_checkpoint_path)
		h , seg1 = sess.run([logits_1, llogits_1], feed_dict={input_p: image[0], flag: True})
		#h , seg2 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[1], flag: True})
		#h , seg3 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[2], flag: True})

	sess.close()
	tf.keras.backend.clear_session()

	mask1 = seg1[...,1]
	mask1 = mask1>0.5
	print(len(image))
	#U = image[0]



	#plt.show()
	mask1 = np.squeeze(mask1)
	print(mask1.shape)
	[H, W, D] = mask1.shape
	xx = np.arange(W)

	xxx = np.tile(xx,[H,1])
	print(xxx.shape)
	yy = np.arange(H)
	yy = yy[::-1]
	yy = np.expand_dims(yy,axis=1)
	#yy = np.fliplr([yy])[0]
	#print(yy)
	yyy = np.tile(yy,(1,W))
	HH = flow.shape[0]
	WW = flow.shape[1]
	h1 = (HH - 160)/2
	w1 = (WW - 96)/2
	print(h1)
	print(w1)
	h1 = np.int(h1)
	w1 = np.int(w1)
	flow = flow[h1:HH-h1,w1:WW-w1,...]
	

	print(np.amin(flow))
	print(flow.shape)
	#yyy = np.fliplr(yyy)
	yyy = yyy+1
	xxx = xxx+1
	print(yyy)
	print(xxx)
	#new_flow = np.zeros(flow.shape)
	print(flow.shape)
	for i in range(flow.shape[2]):
		imaFlow = np.squeeze(flow[...,i,:,:])
		#print(imaFlow.shape)
		statMask = mask1[...,i]
		statMask = statMask.astype(int)
		imaFlow1 = np.zeros(imaFlow.shape)
		#statMask = statMask.astype(np.float32)
		for k in range(imaFlow.shape[2]):
			tt = flow.shape[4]
			phi, alpha, beta = lsqf2(imaFlow[...,k,tt-1],statMask,xxx,yyy)
			phi = phi.astype(np.float32)
			alpha = alpha.astype(np.float32)
			beta = beta.astype(np.float32)
			fitPlane = phi + alpha*xxx + beta*yyy
			zeroMask = imaFlow[:,:,k,tt-1] != 0
			zeroPlane = np.multiply(fitPlane, zeroMask)
			#print(zeroPlane.shape)
			zeroPlane = np.expand_dims(zeroPlane,axis=2)
			factor3D = np.tile(zeroPlane, (1,1, tt))
			imaFlow1[:,:,k,:] = np.squeeze(imaFlow[:,:,k,:]) - factor3D
			#print(phi)
			#print(alpha)
			#print(beta)
			#print(Md)
		flow[:,:,i,:,:] = imaFlow1
	del imaFlow
	del imaFlow1
	print("Finished flow loop in Eddy Current")
	
	del mag
		
			

	io.savemat('new_vel.mat',{'data':flow})
	io.savemat('eddy_mask.mat',{'data':mask1})
	return image, flow
	
