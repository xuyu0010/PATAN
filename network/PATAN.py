import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.init import xavier_uniform_, normal, constant_

try:
	from . import initializer
	from . import TRNmodule
	from .util import load_state
	from .util import BasicBlock, Bottleneck, make_res_layer, SimpleSpatialModule, ClsHead
	from .util import StaticGradReverse as StaticGRL
	from .util import AdversarialNetwork as AdvNet
	from .util import DannDomainNetwork as DannNet
except:
	import initializer
	import TRNmodule
	from util import load_state
	from util import BasicBlock, Bottleneck, make_res_layer, SimpleSpatialModule, ClsHead
	from util import StaticGradReverse as StaticGRL
	from util import AdversarialNetwork as AdvNet
	from util import DannDomainNetwork as DannNet


# @BACKBONES.register_module
class PATAN(nn.Module):
	"""ResNet backbone for spatial stream in TSN. See Args docs in TSN_base.py or TSN.py
	"""

	arch_settings = {
		18: (BasicBlock, (2, 2, 2, 2)),
		34: (BasicBlock, (3, 4, 6, 3)),
		50: (Bottleneck, (3, 4, 6, 3)),
		101: (Bottleneck, (3, 4, 23, 3)),
		152: (Bottleneck, (3, 8, 36, 3))
	}

	def __init__(self,
				 depth=50,
				 pretrained=None, 
				 num_stages=4, 
				 strides=(1, 2, 2, 2), 
				 dilations=(1, 1, 1, 1), 
				 out_indices=(0, 1, 2, 3), 
				 style='pytorch', 
				 frozen_stages=-1,
				 bn_eval=True, 
				 bn_frozen=False, 
				 partial_bn=False, 
				 with_cp=False, 
				 segments=3,
				 consensus_type='trn-m',
				 da_method=None,
				 dynamic_reverse=False,
				 start_clip_weight=0,
				 batch_size=16,
				 num_classes=400,
				 use_spatial=False,):
		super(PATAN, self).__init__()
		if depth not in self.arch_settings:
			raise KeyError('invalid depth {} for resnet'.format(depth))
		self.depth = depth
		self.pretrained = pretrained
		self.num_stages = num_stages
		assert num_stages >= 1 and num_stages <= 4
		self.strides = strides
		self.dilations = dilations
		assert len(strides) == len(dilations) == num_stages
		self.out_indices = out_indices
		assert max(out_indices) < num_stages
		self.style = style
		self.frozen_stages = frozen_stages
		self.bn_eval = bn_eval
		self.bn_frozen = bn_frozen
		self.partial_bn = partial_bn
		self.with_cp = with_cp
		self.segments = segments
		self.consensus_type = consensus_type
		self.da_method = da_method
		self.dynamic_reverse = dynamic_reverse
		self.shared_clip_gf = True
		self.use_spatial = use_spatial

		self.block, stage_blocks = self.arch_settings[depth]
		self.stage_blocks = stage_blocks[:num_stages]
		self.inplanes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		logging.info('TSNNetwork:: number of classes: {}'.format(num_classes))

		self.res_layers = []
		for i, num_blocks in enumerate(self.stage_blocks):
			stride = strides[i]
			dilation = dilations[i]
			planes = 64 * 2**i
			res_layer = make_res_layer(
				self.block,
				self.inplanes,
				planes,
				num_blocks,
				stride=stride,
				dilation=dilation,
				style=self.style,
				with_cp=with_cp)
			self.inplanes = planes * self.block.expansion
			layer_name = 'layer{}'.format(i + 1)
			self.add_module(layer_name, res_layer)
			self.res_layers.append(layer_name)

		self.feat_dim = self.block.expansion * 64 * 2**(len(self.stage_blocks) - 1)
		self.avgpool = SimpleSpatialModule(spatial_type='avg', spatial_size=7)
		logging.info('TSNNetwork:: Utilizing consensus type {}'.format(self.consensus_type))
		self.num_bottleneck = 512
		self.img_feature_dim = 1024
		self.new_fc = nn.Conv2d(2048, self.img_feature_dim, 1)
		self.consensus = TRNmodule.RelationModuleMultiScale(self.img_feature_dim, self.num_bottleneck, self.segments, rand_relation_sample=False)
		self.cls_head = ClsHead(with_avg_pool=False, temp_feat_size=1, sp_feat_size=1, dp_ratio=0.5, in_channels=self.num_bottleneck, num_classes=num_classes)
		self.clip_cls_head = nn.Sequential(
			nn.Conv2d(self.num_bottleneck, self.num_bottleneck, 1),
			ClsHead(with_avg_pool=False, temp_feat_size=1, sp_feat_size=1, dp_ratio=0.5, in_channels=self.num_bottleneck, num_classes=num_classes),
			)
		xavier_uniform_(self.new_fc.weight)
		constant_(self.new_fc.bias, 0)
		if self.use_spatial:
			self.sp_pred_weight = 0.5
			self.spatial_fc = nn.Conv2d(self.img_feature_dim, self.img_feature_dim, 1)
			self.spatial_cls_head = ClsHead(with_avg_pool=False, temp_feat_size=1, sp_feat_size=1, dp_ratio=0.5, in_channels=self.img_feature_dim, num_classes=num_classes)
			xavier_uniform_(self.spatial_fc.weight)
			constant_(self.spatial_fc.bias, 0)

		if self.da_method == 'PATA':
			self.ad_net = AdvNet(self.num_bottleneck, self.num_bottleneck)		
			if self.use_spatial:
				self.spatial_ad_net = AdvNet(self.img_feature_dim, self.img_feature_dim)

			logging.info('TSNNetwork:: Utilizing dynamic reverse layer: {}'.format(self.dynamic_reverse))	
			if self.dynamic_reverse:
				self.high_value = 0.5
				self.pass_num = 0
				self.grl = DynamicGRL()
			else:
				self.reverse_beta = 0.5
				logging.info('TSNNetwork:: Utilizing static reverse layer, degree of reverse is {}'.format(self.reverse_beta))
				self.grl = StaticGRL()

		if self.da_method == 'DANN':
			self.ad_net = DannNet(self.num_bottleneck)
			self.reverse_beta = 0.5
			logging.info('TSNNetwork:: DANN would only utilize static reverse layer, degree of reverse is {}'.format(self.reverse_beta))
			self.grl = StaticGRL()

		self.update_start = start_clip_weight
		self.batch_size = batch_size
		self.TRN_softmax = 0.5
		self.num_classes = num_classes

		self.softmax = nn.Softmax(dim=1)
		self.logsoftmax = nn.LogSoftmax(dim=1)
		self.first_update = True
		self.max_steps = 4200

		#############
		# Initialization

		# initializer.xavier(net=self)
		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/tsn2d_rgb_r50.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self, pretrained['state_dict'])
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, src_x, tgt_x, step):

		src_outs = []
		tgt_outs = []
		src_scale_preds = []
		tgt_scale_preds = []
		src_scale_preds_raw = []
		tgt_scale_preds_raw = []
		src_clip_ws = torch.ones(src_x.shape[0], self.segments-1)
		tgt_clip_ws = torch.ones(src_x.shape[0], self.segments-1)
		if torch.cuda.is_available():
			src_clip_ws = src_clip_ws.cuda()
			tgt_clip_ws = tgt_clip_ws.cuda()

		assert src_x.shape[2] == self.segments, ValueError("source input shape {} not match segments {}".format(src_x.shape[2], self.segments))
		assert tgt_x.shape[2] == self.segments, ValueError("target input shape {} not match segments {}".format(tgt_x.shape[2], self.segments))
		assert self.consensus_type == 'trn-m', NotImplementedError()

		for i in range(src_x.shape[2]):
			src_out = src_x[:, :, 0, :, :]
			tgt_out = tgt_x[:, :, 0, :, :]
			src_out = self.conv1(src_out)
			tgt_out = self.conv1(tgt_out)
			src_out = self.bn1(src_out)
			tgt_out = self.bn1(tgt_out)
			src_out = self.relu(src_out)
			tgt_out = self.relu(tgt_out)
			src_out = self.maxpool(src_out)
			tgt_out = self.maxpool(tgt_out)

			for i, layer_name in enumerate(self.res_layers):
				res_layer = getattr(self, layer_name)
				src_out = res_layer(src_out)
				tgt_out = res_layer(tgt_out)
			src_out = self.avgpool(src_out)
			tgt_out = self.avgpool(tgt_out)
			if 'trn' in self.consensus_type:
				src_out = self.new_fc(src_out)
				tgt_out = self.new_fc(tgt_out)
			src_outs.append(src_out)
			tgt_outs.append(tgt_out)

		src_x = torch.stack(src_outs, dim=2)
		tgt_x = torch.stack(tgt_outs, dim=2)
		if self.use_spatial:
			src_spatial = self.spatial_fc(torch.mean(src_x, 2))
			tgt_spatial = self.spatial_fc(torch.mean(tgt_x, 2))
			src_spatial_pred = self.sp_pred_weight * self.spatial_cls_head(src_spatial)
			tgt_spatial_pred = self.sp_pred_weight * self.spatial_cls_head(tgt_spatial)
		else:
			src_spatial_pred = torch.zeros(src_x.shape[0], self.num_classes)
			tgt_spatial_pred = torch.zeros(tgt_x.shape[0], self.num_classes)
			if torch.cuda.is_available():
				src_spatial_pred = src_spatial_pred.cuda()
				tgt_spatial_pred = tgt_spatial_pred.cuda()

		src_x = src_x.permute(0, 2, 1, 3, 4)
		tgt_x = tgt_x.permute(0, 2, 1, 3, 4)
		_, src_x_scales = self.consensus(src_x.contiguous())
		_, tgt_x_scales = self.consensus(tgt_x.contiguous())

		for scale, src_x_scale in enumerate(src_x_scales):
			if (not step > self.update_start) or (self.update_start > self.max_steps):
				src_scale_pred = self.clip_cls_head(src_x_scale.unsqueeze(-1).unsqueeze(-1).detach())
				tgt_scale_pred = torch.zeros(tgt_clip_ws.shape[0], self.num_classes)
				src_scale_pred_raw = src_scale_pred
				tgt_scale_pred_raw = tgt_scale_pred
				if torch.cuda.is_available():
					src_scale_pred = src_scale_pred.cuda()
					tgt_scale_pred = tgt_scale_pred.cuda()
					src_scale_pred_raw = src_scale_pred_raw.cuda()
					tgt_scale_pred_raw = tgt_scale_pred_raw.cuda()
			else:
				src_scale_pred_raw = self.clip_cls_head(src_x_scale.unsqueeze(-1).unsqueeze(-1).detach())
				tgt_scale_pred_raw = self.clip_cls_head(tgt_x_scales[scale].unsqueeze(-1).unsqueeze(-1).detach())
				src_scale_pred = self.softmax(self.TRN_softmax * src_scale_pred_raw)
				tgt_scale_pred = self.softmax(self.TRN_softmax * tgt_scale_pred_raw)
				src_entropy = torch.sum(-self.softmax(src_scale_pred) * self.logsoftmax(src_scale_pred), 1)
				tgt_entropy = torch.sum(-self.softmax(tgt_scale_pred) * self.logsoftmax(tgt_scale_pred), 1)
				src_clip_w = 1 - src_entropy
				tgt_clip_w = 1 - tgt_entropy
				src_clip_w = src_clip_w.unsqueeze(1)
				tgt_clip_w = tgt_clip_w.unsqueeze(1)
				if scale == 0:
					src_clip_ws = src_clip_w
					tgt_clip_ws = tgt_clip_w
				else:
					src_clip_ws = torch.cat((src_clip_ws, src_clip_w), 1)
					tgt_clip_ws = torch.cat((tgt_clip_ws, tgt_clip_w), 1)

			src_scale_preds.append(src_scale_pred)
			tgt_scale_preds.append(tgt_scale_pred)
			src_scale_preds_raw.append(src_scale_pred_raw)
			tgt_scale_preds_raw.append(tgt_scale_pred_raw)

		src_clip_ws = src_clip_ws / (torch.mean(src_clip_ws, 1).unsqueeze(1).expand(-1, len(src_x_scales)))
		tgt_clip_ws = tgt_clip_ws / (torch.mean(tgt_clip_ws, 1).unsqueeze(1).expand(-1, len(tgt_x_scales)))
		src_clip_ws = src_clip_ws.detach()
		tgt_clip_ws = tgt_clip_ws.detach()
		for i, _ in enumerate(src_scale_preds):
			src_scale_preds[i] = src_scale_preds[i] * (src_clip_ws[:, i].unsqueeze(1).expand(-1, src_scale_preds[i].shape[-1]))
			tgt_scale_preds[i] = tgt_scale_preds[i] * (tgt_clip_ws[:, i].unsqueeze(1).expand(-1, tgt_scale_preds[i].shape[-1]))

		src_x = torch.stack(src_x_scales, 1)
		tgt_x = torch.stack(tgt_x_scales, 1)
		src_x = src_x * (src_clip_ws.unsqueeze(-1).expand(src_x.shape[0], src_x.shape[1], src_x.shape[2]))
		tgt_x = tgt_x * (tgt_clip_ws.unsqueeze(-1).expand(tgt_x.shape[0], tgt_x.shape[1], tgt_x.shape[2]))

		src_x = torch.sum(src_x, 1)
		src_x = src_x.unsqueeze(-1).unsqueeze(-1)
		tgt_x = torch.sum(tgt_x, 1)
		tgt_x = tgt_x.unsqueeze(-1).unsqueeze(-1)

		src_pred = self.cls_head(src_x)
		tgt_pred = self.cls_head(tgt_x)

		src_x = src_x.squeeze(-1).squeeze(-1)
		tgt_x = tgt_x.squeeze(-1).squeeze(-1)

		if 'PATA' in self.da_method.upper():
			if self.dynamic_reverse:
				self.pass_num += 1
				src_adv = self.grl.apply(src_x, self.high_value, self.pass_num)
				tgt_adv = self.grl.apply(tgt_x, self.high_value, self.pass_num)
				if self.use_spatial:
					src_spatial = src_spatial.squeeze(-1).squeeze(-1)
					src_spatial_adv = self.grl.apply(src_spatial, self.high_value, self.pass_num)
					tgt_spatial = tgt_spatial.squeeze(-1).squeeze(-1)
					tgt_spatial_adv = self.grl.apply(tgt_spatial, self.high_value, self.pass_num)
			else:
				src_adv = self.grl.apply(src_x, self.reverse_beta)
				tgt_adv = self.grl.apply(tgt_x, self.reverse_beta)				
				if self.use_spatial:
					src_spatial = src_spatial.squeeze(-1).squeeze(-1)
					src_spatial_adv = self.grl.apply(src_spatial, self.reverse_beta)
					tgt_spatial = tgt_spatial.squeeze(-1).squeeze(-1)
					tgt_spatial_adv = self.grl.apply(tgt_spatial, self.reverse_beta)

			src_adv = self.ad_net(src_adv)
			tgt_adv = self.ad_net(tgt_adv)
			if self.use_spatial:
				src_spatial_adv = self.spatial_ad_net(src_spatial_adv)
				tgt_spatial_adv = self.spatial_ad_net(tgt_spatial_adv)
			else:
				src_spatial_adv = torch.zeros(src_x.shape[0], 1)
				tgt_spatial_adv = torch.zeros(tgt_x.shape[0], 1)
				if torch.cuda.is_available():
					src_spatial_adv = src_spatial_adv.cuda()
					tgt_spatial_adv = tgt_spatial_adv.cuda()

		else:
			src_adv = torch.zeros(src_x.shape[0], 1)
			tgt_adv = torch.zeros(tgt_x.shape[0], 1)
			src_spatial_adv = torch.zeros(src_x.shape[0], 1)
			tgt_spatial_adv = torch.zeros(tgt_x.shape[0], 1)
			if torch.cuda.is_available():
				src_adv = src_adv.cuda()
				tgt_adv = tgt_adv.cuda()
				src_spatial_adv = src_spatial_adv.cuda()
				tgt_spatial_adv = tgt_spatial_adv.cuda()

		return src_x, tgt_x, src_pred, tgt_pred, src_spatial_pred, tgt_spatial_pred,\
				src_scale_preds, tgt_scale_preds, src_scale_preds_raw, tgt_scale_preds_raw,\
				src_adv, tgt_adv, src_spatial_adv, tgt_spatial_adv


if __name__ == "__main__":

	net = PATAN(pretrained=False,segments=5,consensus_type='trn-m',da_method='PATA',dynamic_reverse=False,start_clip_weight=3,num_classes=7,batch_size=2,use_spatial=False)
	if torch.cuda.is_available():
		net = net.cuda()
	src_data = torch.randn(2,3,5,224,224) # [bs,c,t,h,w]
	tgt_data = torch.randn(2,3,5,224,224) # [bs,c,t,h,w]
	if torch.cuda.is_available():
		src_data = src_data.cuda()
		tgt_data = tgt_data.cuda()
	src_ft,tgt_ft,src_pred,tgt_pred,src_sp_pred,tgt_sp_pred,src_preds,tgt_preds,src_preds_raw,tgt_preds_raw,src_adv,tgt_adv,src_sp_adv,tgt_sp_adv = net(src_data, tgt_data, 0)
