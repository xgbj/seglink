import tensorflow as tf

slim = tf.contrib.slim

def _upscore_layer(self, layer, target_layer):
#             target_shape = target_layer.shape[1:-1] # NHWC
		target_shape = tf.shape(target_layer)[1:-1]
		upscored = tf.image.resize_images(layer, target_shape)
		return upscored

def basenet(inputs):
		"""
		backbone net of vgg16
		"""
		# End_points collect relevant activations for external use.
		end_points = {}
		# Original VGG-16 blocks.
		with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
			net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			end_points['conv1_2'] = net
			net = slim.max_pool2d(net, [2, 2], scope='pool1')
			# Block 2.
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			end_points['conv2_2'] = net
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			# Block 3.
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
			end_points['conv3_3'] = net
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			# Block 4.
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
			end_points['conv4_3'] = net
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			# Block 5.
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
			end_points['conv5_3'] = net
			net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

			# fc6 as conv, dilation is added
			net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='fc6')
			fc6_layer = net
			end_points['fc6'] = net

			# fc7 as conv
			net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
			fc7_layer = net
			#end_points['fc7'] = net

			# Additional SSD blocks.
			# conv6/7/8/9/10: 1x1 and 3x3 convolutions stride 2 (except lasts).
			net = slim.conv2d(net, 256, [1, 1], scope='conv6_1')
			net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv6_2', padding='SAME')
			conv6_2 = net
			#end_points['conv6_2'] = net

			net = slim.conv2d(net, 128, [1, 1], scope='conv7_1')
			net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv7_2', padding='SAME')
			conv7_2 = net
			#end_points['conv7_2'] = net

			net = slim.conv2d(net, 128, [1, 1], scope='conv8_1')
			net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv8_2', padding='SAME')
			conv8_2 = net
			#end_points['conv8_2'] = net

			net = slim.conv2d(net, 128, [1, 1], scope='conv9_1')
			net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_2', padding='SAME')
			conv9_2 = net
			end_points['conv9_2'] = conv9_2

			#upsampling layer
			upscore9_2 = _upscore_layer(conv9_2,conv8_2)
			combine8_9 = tf.concat([conv8_2, upscore9_2], axis = 0)
			combine8_9 = slim.conv2d(combine8_9, 256, [1, 1], scope='combine8_9', padding='SAME')
			end_points['conv8_2'] = combine8_9

			upscore8_2 = _upscore_layer(combine8_9, conv7_2)
			combine7_8 = tf.concat([conv7_2, upscore8_2], axis=0)
			combine7_8 = slim.conv2d(combine7_8, 256, [1, 1], scope='combine7_8', padding='SAME')
			end_points['conv7_2'] = combine7_8

			upscore7_2 = _upscore_layer(combine7_8, conv6_2)
			combine6_7 = tf.concat([conv6_2, upscore7_2], axis=0)
			combine6_7 = slim.conv2d(combine6_7, 256, [1, 1], scope='combine6_7', padding='SAME')
			end_points['conv6_2'] = combine6_7

			upscore6_2 = _upscore_layer(combine6_7, fc7_layer)
			combine_fc7 = tf.concat([fc7_layer, upscore6_2], axis=0)
			combine_fc7 = slim.conv2d(combine_fc7, 512, [1, 1], scope='combine_fc7', padding='SAME')
			end_points['fc7'] = combine_fc7

			conv4_3 = end_points['conv4_3']
			upscore_fc7 = _upscore_layer(combine_fc7, conv4_3)
			combine_conv4 = tf.concat([conv4_3, upscore_fc7], axis=0)
			combine_conv4 = slim.conv2d(combine_conv4, 512, [1, 1], scope='combine_conv4', padding='SAME')
			end_points['conv4_3'] = combine_conv4

		return net, end_points;


