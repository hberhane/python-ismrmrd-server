import tensorflow as tf
#tf.reset_default_graph()

def encoder_layer(x_con, channels, name,training, pool=True):
    
    with tf.name_scope("encoder_block_{}".format(name)):
        for i in range(channels):
            
            x = tf.layers.conv3d(x_con,12,kernel_size=[3,3,3],padding='SAME')
            x = tf.layers.dropout(x,rate=0.1,training=training)
            x = tf.layers.batch_normalization(x,training=training,renorm=True)
            x = tf.nn.relu(x)
            x_con = tf.concat([x,x_con], axis = 4)
        if pool is False:
            return x_con
        
        #x = tf.layers.conv3d(x_con,12,kernel_size=[1,1,1],padding='SAME')
        #x = tf.layers.dropout(x,rate=0.1,training=training)
        #x = tf.layers.batch_normalization(x,training=training,renorm=True)
        #x = tf.nn.relu(x)
        pool = tf.layers.max_pooling3d(x_con,pool_size = [2,2,1], strides=[2,2,1],data_format='channels_last')

        return x_con, pool
def decoder_layer(input_, x, ch, name, upscale = [2,2,2]):
        
    up = tf.layers.conv3d_transpose(input_,filters=12,kernel_size = [2,2,1],strides = [2,2,1],padding='SAME',name='upsample'+str(name), use_bias=False)
    up = tf.concat([up,x], axis=-1, name='merge'+str(name))
    return up
class Unet():
    def __init__(self, x, training):
        #self.filters = filters
        self.training = training
        self.model = self.U_net(x)

    
    def U_net(self,input_):
        skip_conn = []


        conv1, pool1 = encoder_layer(input_,channels=2,name="encode_"+str(1),training=self.training, pool=True)
        conv2, pool2 = encoder_layer(pool1,channels=4,name="encode_"+str(2),training=self.training, pool=True)
        conv3, pool3 = encoder_layer(pool2,channels=6,name="encode_"+str(3),training=self.training, pool=True)
        conv4, pool4 = encoder_layer(pool3,channels=8,name="encode_"+str(4),training=self.training, pool=True)
        conv5, pool5 = encoder_layer(pool4,channels=10,name="encode_"+str(5),training=self.training, pool=True)
        conv6 = encoder_layer(pool5,channels=12,name="encode_"+str(5),training=self.training, pool=False)
        up1 = decoder_layer(conv6,conv5,10,name=1)
        conv7 = encoder_layer(up1,channels=10,name="conv"+str(6),training=self.training, pool=False)
        up2 = decoder_layer(conv7,conv4,8,name=2)
        conv8 = encoder_layer(up2,channels=8,name="encode_"+str(7),training=self.training, pool=False)
        up3 = decoder_layer(conv8,conv3,6,name=3)
        conv9 = encoder_layer(up3,channels=6,name="encode_"+str(8),training=self.training, pool=False)
        up4 = decoder_layer(conv9,conv2,4,name=4)
        conv10 = encoder_layer(up4,channels= 4,name="encode_"+str(9),training=self.training, pool=False)
        up5 = decoder_layer(conv10,conv1,2,name=5)
        conv11 = encoder_layer(up5,channels= 2,name="encode_"+str(10),training=self.training, pool=False)



        score = tf.layers.conv3d(conv11,2,(1,1,1),name='logits',padding='SAME')
        #tf.get_variable_scope().reuse_variables()
        return score