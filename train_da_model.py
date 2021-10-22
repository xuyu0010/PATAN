import os
import logging

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from train import metric
# from train.da_model import da_model
from train.da_model_test import da_model
from data import da_iterator_factory as da_fac
from train.lr_scheduler import MultiFactorScheduler as MFS


def train_da_model(sym_net, model_prefix, da_method, num_classes, src_dataset, tgt_dataset, input_conf, use_spatial,
				segments=3, frame_per_seg=1, train_frame_interval=2, val_frame_interval=2,
				resume_epoch=-1, batch_size=4, save_freq=1, step_class_w_update=50, start_clip_weight=0,
				lr_base=0.01, lr_factor=0.1, lr_steps=[400000, 800000], end_epoch=1000, fine_tune=False, **kwargs):

	assert torch.cuda.is_available(), "Currently, we only support CUDA version"
	torch.multiprocessing.set_sharing_strategy('file_system')

	# data iterator
	arid_mean = [0.079612, 0.073888, 0.072454]
	arid_std = [0.100459, 0.09705, 0.089911]
	src_mean = tgt_mean = input_conf['mean']
	src_std = tgt_std = input_conf['std']
	if src_dataset == 'ARID':
		src_mean = arid_mean
		src_std = arid_std
	if tgt_dataset == 'ARID':
		tgt_mean = arid_mean
		tgt_std = arid_std
	iter_seed = torch.initial_seed() + 100 + max(0, resume_epoch) * 100
	src_train_iter, src_eval_iter = da_fac.creat(name=src_dataset, batch_size=batch_size, segments=segments, frame_per_seg=frame_per_seg, 
										train_interval=train_frame_interval, val_interval=val_frame_interval, mean=src_mean, std=src_std, seed=iter_seed)
	tgt_train_iter, tgt_eval_iter = da_fac.creat(name=tgt_dataset, batch_size=batch_size, segments=segments, frame_per_seg=frame_per_seg, 
										train_interval=train_frame_interval, val_interval=val_frame_interval, mean=tgt_mean, std=tgt_std, seed=iter_seed)

	# wapper (dynamic model)
	net = da_model(net=sym_net,model_prefix=model_prefix,da_method=da_method,num_classes=num_classes,use_spatial=use_spatial,
				step_callback_freq=50,step_class_w_update=step_class_w_update,start_clip_weight=start_clip_weight,save_freq=save_freq,batch_size=batch_size,)
	net.net.cuda()

	# config optimization
	param_base_layers = []
	param_class_layers = []
	param_adv_layers = []
	param_clip_class_layers = []
	name_base_layers = []
	for name, param in net.net.named_parameters():
		if fine_tune:
			if name.startswith('clip'):
				param_clip_class_layers.append(param)
			elif 'classifier' in name or 'fc' in name or 'cls' in name:
				param_class_layers.append(param)
			elif 'ad' in name:
				param_adv_layers.append(param)
			else:
				param_base_layers.append(param)
				name_base_layers.append(name)
		else:
			param_class_layers.append(param)

	if name_base_layers:
		out = "[\'" + '\', \''.join(name_base_layers) + "\']"
		logging.info("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers), out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

	net.net = torch.nn.DataParallel(net.net).cuda()

	optimizer = torch.optim.SGD([{'params':param_base_layers,'lr_mult':0.1}, {'params':param_adv_layers,'lr_mult':1.0},
								 {'params':param_class_layers,'lr_mult':1.0}, {'params':param_clip_class_layers,'lr_mult':1.0}],
								lr=lr_base,momentum=0.9,weight_decay=0.0001,nesterov=True)

	# resume training: model and optimizer
	if resume_epoch < 0:
		epoch_start = 0
		step_counter = 0
	else:
		net.load_checkpoint(epoch=resume_epoch, optimizer=optimizer)
		epoch_start = resume_epoch
		step_counter = epoch_start * src_train_iter.__len__()

	# set learning rate scheduler
	num_worker = 1
	lr_scheduler = MFS(base_lr=lr_base, steps=[int(x/(batch_size*num_worker)) for x in lr_steps], factor=lr_factor, step_counter=step_counter)
	
	# define evaluation metric
	metrics_src = metric.MetricList(metric.Loss(name="loss-ce"), metric.Accuracy(name="top1", topk=1), metric.Accuracy(name="top5", topk=5),)
	metrics_tgt = metric.MetricList(metric.Loss(name="loss-ce"), metric.Accuracy(name="top1", topk=1), metric.Accuracy(name="top5", topk=5),)
	# enable cudnn tune
	# cudnn.benchmark = True

	net.fit(src_train_iter=src_train_iter, tgt_train_iter=tgt_train_iter, tgt_eval_iter=tgt_eval_iter, optimizer=optimizer, lr_scheduler=lr_scheduler, 
			metrics_src=metrics_src, metrics_tgt=metrics_tgt, epoch_start=epoch_start, epoch_end=end_epoch,)
