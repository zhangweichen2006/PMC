import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxPool3dSamePadding(nn.MaxPool3d):
	
	def compute_pad(self, dim, s):
		if s % self.stride[dim] == 0:
			return max(self.kernel_size[dim] - self.stride[dim], 0)
		else:
			return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

	def forward(self, x):
		# compute 'same' padding
		(batch, channel, t, h, w) = x.size()
		#print t,h,w
		out_t = np.ceil(float(t) / float(self.stride[0]))
		out_h = np.ceil(float(h) / float(self.stride[1]))
		out_w = np.ceil(float(w) / float(self.stride[2]))
		#print out_t, out_h, out_w
		pad_t = self.compute_pad(0, t)
		pad_h = self.compute_pad(1, h)
		pad_w = self.compute_pad(2, w)
		#print pad_t, pad_h, pad_w

		pad_t_f = pad_t // 2
		pad_t_b = pad_t - pad_t_f
		pad_h_f = pad_h // 2
		pad_h_b = pad_h - pad_h_f
		pad_w_f = pad_w // 2
		pad_w_b = pad_w - pad_w_f

		pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
		#print x.size()
		#print pad
		x = F.pad(x, pad)
		return super(MaxPool3dSamePadding, self).forward(x)
	

class Unit3D(nn.Module):

	def __init__(self, in_channels,
				 output_channels,
				 kernel_shape=(1, 1, 1),
				 stride=(1, 1, 1),
				 padding=0,
				 activation_fn=F.relu,
				 use_batch_norm=True,
				 use_bias=True,
				 name='unit_3d'):
		
		"""Initializes Unit3D module."""
		super(Unit3D, self).__init__()
		
		self._output_channels = output_channels
		self._kernel_shape = kernel_shape
		self._stride = stride
		self._use_batch_norm = use_batch_norm
		self._activation_fn = activation_fn
		self._use_bias = use_bias
		self.name = name
		self.padding = padding
		
		self.conv3d = nn.Conv3d(in_channels=in_channels,
								out_channels=self._output_channels,
								kernel_size=self._kernel_shape,
								stride=self._stride,
								padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
								bias=self._use_bias)
		
		if self._use_batch_norm:
			self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01, track_running_stats=False)

	def compute_pad(self, dim, s):
		if s % self._stride[dim] == 0:
			return max(self._kernel_shape[dim] - self._stride[dim], 0)
		else:
			return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

			
	def forward(self, x):
		# compute 'same' padding
		(batch, channel, t, h, w) = x.size()
		#print t,h,w
		out_t = np.ceil(float(t) / float(self._stride[0]))
		out_h = np.ceil(float(h) / float(self._stride[1]))
		out_w = np.ceil(float(w) / float(self._stride[2]))
		#print out_t, out_h, out_w
		pad_t = self.compute_pad(0, t)
		pad_h = self.compute_pad(1, h)
		pad_w = self.compute_pad(2, w)
		#print pad_t, pad_h, pad_w

		pad_t_f = pad_t // 2
		pad_t_b = pad_t - pad_t_f
		pad_h_f = pad_h // 2
		pad_h_b = pad_h - pad_h_f
		pad_w_f = pad_w // 2
		pad_w_b = pad_w - pad_w_f

		pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
		#print x.size()
		#print pad
		x = F.pad(x, pad)
		#print x.size()        

		x = self.conv3d(x)
		if self._use_batch_norm:
			x = self.bn(x)
		if self._activation_fn is not None:
			x = self._activation_fn(x)
		return x

class self_gating(nn.Module):
	def __init__(self, num_channels):
		super(self_gating, self).__init__()
		self.conv3d = nn.Conv3d(num_channels, num_channels, kernel_size=1, bias=False)

	def forward(self, input_tensor):

		spatiotemporal_average = input_tensor.mean(2, True).mean(3, True).mean(4, True)

		weights = self.conv3d(spatiotemporal_average)

		weights = weights.expand_as(input_tensor)

		weights = F.sigmoid(weights)

		return torch.mul(input_tensor, weights)


class conv3d_spatiotemporal(nn.Module):
	def __init__(self, in_channel,
				 out_channel,
				 kernel_size,
				 stride=(1,1,1),
				 padding=0,
				 separable=True,
				 name='conv3d_spatiotemporal',
				 use_gating=False):
		super(conv3d_spatiotemporal, self).__init__()
		self.name = name
		self.kernel_size = kernel_size
		self.separable = separable
		self.use_gating = use_gating
		if self.separable and kernel_size != 1:
			spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
			temporal_kernel_size = (kernel_size[0], 1, 1)
			if isinstance(stride, tuple) and len(stride) == 3:
				spatial_stride = (1, stride[1], stride[2])
				temporal_stride = (stride[0], 1, 1)
			else:
				spatial_stride = [1, stride, stride]
				temporal_stride = [stride, 1, 1]
			self.conv3d_S = Unit3D(in_channel, out_channel, kernel_shape=spatial_kernel_size, padding=padding, stride=spatial_stride, name=name, use_bias=False)
			self.conv3d_T = Unit3D(out_channel, out_channel, kernel_shape=temporal_kernel_size, padding=padding, stride=temporal_stride, name=name+'/temporal', use_batch_norm=False)
		else:
			self.conv3d = Unit3D(in_channel, out_channel, kernel_shape=kernel_size, padding=padding, stride=stride, name=name, use_bias=False)
		if use_gating:
			self.gating = self_gating(out_channel)

	def forward(self, inputs):
		if self.separable and self.kernel_size != 1:
			out = self.conv3d_S(inputs)
			out = self.conv3d_T(out)
		else:
			out = self.conv3d(inputs)
		if self.use_gating:
			out = self.gating(out)
		return out

class inception_block_v1_3d(nn.Module):
	def __init__(self, in_channel,
				 out_channel_0_0a,
				 out_channel_1_0a,
				 out_channel_1_0b,
				 out_channel_2_0a,
				 out_channel_2_0b,
				 out_channel_3_0b,
				 temporal_kernel_size=3,
				 use_gating=True,
				 name='inception_block_v1_3d'):
		super(inception_block_v1_3d, self).__init__()
		self.name = name
		self.use_gating = use_gating
		if use_gating:
			self.gating_branch0 = self_gating(out_channel_0_0a)
			self.gating_branch1 = self_gating(out_channel_1_0b)
			self.gating_branch2 = self_gating(out_channel_2_0b)
			self.gating_branch3 = self_gating(out_channel_3_0b)
		self.branch0_0a = Unit3D(in_channel, out_channel_0_0a, kernel_shape=(1,1,1), name=name+'/Branch_0/Conv3d_0a_1x1', use_bias=False)
		self.branch1_0a = Unit3D(in_channel, out_channel_1_0a, kernel_shape=(1,1,1), name=name+'/Branch_1/Conv3d_0a_1x1', use_bias=False)
		self.branch2_0a = Unit3D(in_channel, out_channel_2_0a, kernel_shape=(1,1,1), name=name+'/Branch_2/Conv3d_0a_1x1', use_bias=False)
		self.branch3_0a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
		self.branch1_0b = conv3d_spatiotemporal(out_channel_1_0a, out_channel_1_0b, (temporal_kernel_size, 3, 3), name=name+'/Branch_1/Conv2d_0b_3x3')
		self.branch2_0b = conv3d_spatiotemporal(out_channel_2_0a, out_channel_2_0b, (temporal_kernel_size, 3, 3), name=name+'/Branch_2/Conv2d_0b_3x3')
		self.branch3_0b = Unit3D(in_channel, out_channel_3_0b, kernel_shape=(1,1,1), name=name+'/Branch_3/Conv2d_0b_1x1', use_bias=False)

	def forward(self, inputs):
		b0 = self.branch0_0a(inputs)
		b1 = self.branch1_0b(self.branch1_0a(inputs))
		b2 = self.branch2_0b(self.branch2_0a(inputs))
		b3 = self.branch3_0b(self.branch3_0a(inputs))
		if self.use_gating:
			b0 = self.gating_branch0(b0)
			b1 = self.gating_branch1(b1)
			b2 = self.gating_branch2(b2)
			b3 = self.gating_branch3(b3)
		return torch.cat([b0,b1,b2,b3], dim=1)

class s3dg_base(nn.Module):
	VALID_ENDPOINTS = (
		'Conv2d_1a_7x7',
		'MaxPool_2a_3x3',
		'Conv2d_2b_1x1',
		'Conv2d_2c_3x3',
		'MaxPool_3a_3x3',
		'Mixed_3b',
		'Mixed_3c',
		'MaxPool_4a_3x3',
		'Mixed_4b',
		'Mixed_4c',
		'Mixed_4d',
		'Mixed_4e',
		'Mixed_4f',
		'MaxPool_5a_2x2',
		'Mixed_5b',
		'Mixed_5c',
		'Logits',
		'Predictions',
	)

	OUT_ENDPOINTS = (
		'Conv2d_2c_3x3',
		'Mixed_3c',
		'Mixed_4f',
		'Mixed_5c',
		)
	def __init__(self, first_temporal_kernel_size=3, 
				 temporal_conv_startat='Conv2d_2c_3x3',
				 in_channels=3,
				 gating_startat='Conv2d_2c_3x3',
				 final_endpoint='Mixed_5c',
				 name='InceptionV1/', dropout_keep_prob=0.5, num_classes=400, spatial_squeeze=True):
		super(s3dg_base, self).__init__()
		self._spatial_squeeze = spatial_squeeze
		self._num_classes = num_classes
		self.end_points = {}
		t = 1
		use_gating = False
		self.final_endpoint = final_endpoint

		end_point = 'Conv2d_1a_7x7'
		self.end_points[end_point] = conv3d_spatiotemporal(in_channels, 64, (first_temporal_kernel_size,7,7),
															stride=(2,2,2), separable=False, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'MaxPool_2a_3x3'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1,3,3), stride=(1,2,2), padding=0)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Conv2d_2b_1x1'
		self.end_points[end_point] = Unit3D(64, 64, (1,1,1), name=name+end_point, use_bias=False)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Conv2d_2c_3x3'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = conv3d_spatiotemporal(64, 192, (t,3,3), name=name+end_point, use_gating=use_gating)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'MaxPool_3a_3x3'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1,3,3), stride=(1,2,2), padding=0)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_3b'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(192, 64, 96, 128, 16, 32, 32, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_3c'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(256, 128, 128, 192, 32, 96, 64, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'MaxPool_4a_3x3'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(3,3,3), stride=(2,2,2), padding=0)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_4b'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(480, 192, 96, 208, 16, 48, 64, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_4c'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(512, 160, 112, 224, 24, 64, 64, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_4d'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(512, 128, 128, 256, 24, 64, 64, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_4e'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(512, 112, 144, 288, 32, 64, 64, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_4f'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(528, 256, 160, 320, 32, 128, 128, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'MaxPool_5a_2x2'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(2,2,2), stride=(2,2,2), padding=0)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Mixed_5b'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(832, 256, 160, 320, 32, 128, 128, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return
			
		end_point = 'Mixed_5c'
		if temporal_conv_startat == end_point:
			t = 3
		if gating_startat == end_point:
			use_gating = True
		self.end_points[end_point] = inception_block_v1_3d(832, 384, 192, 384, 48, 128, 128, t, use_gating, name=name+end_point)
		if self.final_endpoint == end_point:
			self.build()
			return

		end_point = 'Logits'
		self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
									 stride=(1, 1, 1))
		self.dropout = nn.Dropout(dropout_keep_prob)
		self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
							 kernel_shape=[1, 1, 1],
							 padding=0,
							 activation_fn=None,
							 use_batch_norm=False,
							 use_bias=True,
							 name='logits')

		self.build()

	def build(self):
		for k in self.end_points.keys():
			self.add_module(k, self.end_points[k])
		
	def forward(self, x):
		outs = []
		for end_point in self.VALID_ENDPOINTS:
			if end_point in self.end_points:
				x = self._modules[end_point](x) # use _modules to work with dataparallel
				if end_point in self.OUT_ENDPOINTS:
					out = x.mean(2)
					# print(end_point,out.size())
					outs.append(out)

		# x = self.logits(self.dropout(self.avg_pool(x)))
		# if self._spatial_squeeze:
		# 	logits = x.squeeze(3).squeeze(3).mean(2)
		# logits is batch X time X classes, which is what we want to work with
		return outs#, logits


class S3DG(nn.Module):
    def __init__(self):
        super(S3DG, self).__init__()
        self.S3DG_rgb = s3dg_base(final_endpoint='Mixed_5c', in_channels=3)
        self.S3DG_flow = s3dg_base(final_endpoint='Mixed_5c', in_channels=2)
        self.init_weight()

    def init_weight(self, pretrain=True):
        if pretrain:
            s3dg_weights = torch.load('s3dg_rgb_600.pt')
            s3dg_base_weights = self.S3DG_rgb.state_dict()
            # print(i3d_base_weights.keys())
            for k, v in s3dg_base_weights.items():
                s3dg_base_weights[k] = s3dg_weights[k]
            self.S3DG_rgb.load_state_dict(s3dg_base_weights)
            s3dg_flow_weights = torch.load('s3dg_flow_600.pt')
            s3dg_flow_base_weights = self.S3DG_flow.state_dict()
            for k, v in s3dg_flow_base_weights.items():
                s3dg_flow_base_weights[k] = s3dg_flow_weights[k]
            self.S3DG_flow.load_state_dict(s3dg_flow_base_weights)

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # print(x.size())
        out_rgb = self.S3DG_rgb(x[:,:20,:,:,:].permute(0,2,1,3,4))
        out_flow = self.S3DG_flow(x[:,20:,:2,:,:].permute(0,2,1,3,4))
        return (out_rgb, out_flow)

    def train(self, mode=True):
        super(S3DG, self).train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for params in m.parameters():
                    params.requires_grad = False
        # if mode and self.frozen_stages >= 0:
        #     for param in self.conv1.parameters():
        #         param.requires_grad = False
        #     for param in self.bn1.parameters():
        #         param.requires_grad = False
        #     self.bn1.eval()
        #     self.bn1.weight.requires_grad = False
        #     self.bn1.bias.requires_grad = False
        #     for i in range(1, self.frozen_stages + 1):
        #         mod = getattr(self, 'layer{}'.format(i))
        #         mod.eval()
        #         for param in mod.parameters():
        #             param.requires_grad = False

