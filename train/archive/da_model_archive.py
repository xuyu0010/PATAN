"""
Author: Yunpeng Chen
"""
import os
import time
import socket
import logging
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import metric
from . import callback

"""
Static Model
"""
class static_model(object):

	def __init__(self, net, model_prefix='', da_method=None, **kwargs):

		if kwargs:
			logging.warning("Unknown kwargs: {}".format(kwargs))

		# init params
		self.net = net
		self.model_prefix = model_prefix
		self.da_method = da_method
		self.pada_tradeoff = 1.0
		self.dann_tradeoff = 1.0

		if self.da_method is not None:
			logging.info("Using domain adaptation method: {}".format(da_method))

	def load_state(self, state_dict, strict=False):
		if strict:
			self.net.load_state_dict(state_dict=state_dict)
		else:
			# customized partialy load function
			net_state_keys = list(self.net.state_dict().keys())
			for name, param in state_dict.items():
				if name in self.net.state_dict().keys():
					dst_param_shape = self.net.state_dict()[name].shape
					if param.shape == dst_param_shape:
						self.net.state_dict()[name].copy_(param.view(dst_param_shape))
						net_state_keys.remove(name)
			# indicating missed keys
			if net_state_keys:
				num_batches_list = []
				for i in range(len(net_state_keys)):
					if 'num_batches_tracked' in net_state_keys[i]:
						num_batches_list.append(net_state_keys[i])
				pruned_additional_states = [x for x in net_state_keys if x not in num_batches_list]
				if pruned_additional_states:
					logging.info("There are layers in current network not initialized by pretrained")
					pruned = "[\'" + '\', \''.join(pruned_additional_states) + "\']"
					logging.warning(">> Failed to load: {}".format(pruned[0:150] + " ... " + pruned[-150:]))
				return False
		return True

	def get_checkpoint_path(self, epoch):
		assert self.model_prefix, "model_prefix undefined!"
		if torch.distributed.is_initialized():
			hostname = socket.gethostname()
			checkpoint_path = "{}_at-{}_ep-{:04d}.pth".format(self.model_prefix, hostname, epoch)
		else:
			checkpoint_path = "{}_ep-{:04d}.pth".format(self.model_prefix, epoch)
		return checkpoint_path

	def load_checkpoint(self, epoch, optimizer=None):

		load_path = self.get_checkpoint_path(epoch)
		assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

		checkpoint = torch.load(load_path)

		all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)

		if optimizer:
			if 'optimizer' in checkpoint.keys() and all_params_matched:
				optimizer.load_state_dict(checkpoint['optimizer'])
				logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
			else:
				logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
		else:
			logging.info("Only model state resumed from: `{}'".format(load_path))

		if 'epoch' in checkpoint.keys():
			if checkpoint['epoch'] != epoch:
				logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

	def save_checkpoint(self, epoch, optimizer_state=None):

		save_path = self.get_checkpoint_path(epoch)
		save_folder = os.path.dirname(save_path)

		if not os.path.exists(save_folder):
			logging.debug("mkdir {}".format(save_folder))
			os.makedirs(save_folder)

		if not optimizer_state:
			torch.save({'epoch': epoch, 'state_dict': self.net.state_dict()}, save_path)
			logging.info("Checkpoint (only model) saved to: {}".format(save_path))
		else:
			torch.save({'epoch': epoch, 'state_dict': self.net.state_dict(), 'optimizer': optimizer_state}, save_path)
			logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))


	def forward(self, src_data, tgt_data, src_label, tgt_label, class_weight):
		""" typical forward function with:
			single output and single loss
		"""
		src_data = src_data.float().cuda()
		tgt_data = tgt_data.float().cuda()
		src_label = src_label.cuda()
		tgt_label = tgt_label.cuda()
		if self.net.training:
			torch.set_grad_enabled(True)
		else:
			torch.set_grad_enabled(False)

		input_all = torch.cat((src_data, tgt_data), dim=0)
		_, pred_all, adv_out_all = self.net(input_all)
		softmax_out_all = nn.Softmax(dim=1)(pred_all).detach()
		# output_src = output_all.narrow(0, 0, int(input_all.size(0)/2))
		# output_tgt = output_all.narrow(0, int(input_all.size(0)/2), int(input_all.size(0)/2))
		# output = output_tgt
		pred_src = pred_all.narrow(0, 0, int(input_all.size(0)/2))
		pred_tgt = pred_all.narrow(0, int(input_all.size(0)/2), int(input_all.size(0)/2))
		output_tgt = pred_tgt
		output_src = pred_src

		if src_label is not None:
			if self.net.training:
				losses = []
				# source data class classification
				class_criterion = nn.CrossEntropyLoss(weight = class_weight)
				if torch.cuda.is_available():
					class_criterion = class_criterion.cuda()
				loss = class_criterion(output_src, src_label)
				# domain classification: 1 for source and 0 for target
				if self.da_method == 'PADA':
					loss_pada = metric.PADA(input_all, adv_out_all, src_label, class_weight)
					loss += self.pada_tradeoff * loss_pada
					losses.append('loss_pada: ' + str(loss_pada.item()))
				elif self.da_method == 'DANN':
					dann_criterion = nn.CrossEntropyLoss()
					dann_criterion = dann_criterion.cuda()
					src_domain_label = torch.ones(src_data.size(0)).long()
					tgt_domain_label = torch.zeros(tgt_data.size(0)).long()
					domain_label = torch.cat((src_domain_label,tgt_domain_label),0)
					if torch.cuda.is_available():
						domain_label = domain_label.cuda(non_blocking=True)
					loss_dann = dann_criterion(adv_out_all, domain_label)
					loss += self.dann_tradeoff * loss_dann
					losses.append('loss_dann: ' + str(loss_dann.item()))
			else:
				losses = []
				class_criterion = nn.CrossEntropyLoss(weight = class_weight)
				if torch.cuda.is_available():
					class_criterion = class_criterion.cuda()
				loss = class_criterion(output_tgt, tgt_label)
		else:
			loss = None
			losses = []
		return [output_tgt], [output_src], [loss], losses


"""
Dynamic model that is able to update itself
"""
class da_model(static_model):

	def __init__(self,net,model_prefix='',da_method=None,num_classes=400,
					step_callback=None,step_callback_freq=50,step_class_w_update=100,epoch_callback=None,save_freq=1,batch_size=None,**kwargs):

		# load parameters
		if kwargs:
			logging.warning("Unknown kwargs in da_model: {}".format(kwargs))

		super(da_model, self).__init__(net,model_prefix=model_prefix,da_method=da_method)

		# load optional arguments
		# - callbacks
		self.callback_kwargs = {'epoch': None, 'batch': None, 'sample_elapse': None, 'update_elapse': None, 'epoch_elapse': None, 'namevals': None, 'optimizer_dict': None,}

		if not step_callback:
			step_callback = callback.CallbackList(callback.SpeedMonitor(), callback.MetricPrinter())

		if not epoch_callback:
			epoch_callback = (lambda **kwargs: None)

		self.num_classes = num_classes
		self.step_callback = step_callback
		self.step_callback_freq = step_callback_freq
		self.step_class_w_update = step_class_w_update
		self.epoch_callback = epoch_callback
		self.save_freq = save_freq
		self.batch_size = batch_size
		self.da_method = da_method


	"""
	In order to customize the callback function,
	you will have to overwrite the functions below
	"""
	def step_end_callback(self):
		# logging.debug("Step {} finished!".format(self.i_step))
		self.step_callback(**(self.callback_kwargs))

	def epoch_end_callback(self):
		self.epoch_callback(**(self.callback_kwargs))

		if self.callback_kwargs['epoch_elapse'] is not None:
			logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
					self.callback_kwargs['epoch'], self.callback_kwargs['epoch_elapse'], self.callback_kwargs['epoch_elapse']/3600.))

		if self.callback_kwargs['epoch'] == 0 or ((self.callback_kwargs['epoch']+1) % self.save_freq) == 0:
			self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1, optimizer_state=self.callback_kwargs['optimizer_dict'])

	"""
	Learning rate
	"""
	def adjust_learning_rate(self, lr, optimizer):
		for param_group in optimizer.param_groups:
			if 'lr_mult' in param_group:
				lr_mult = param_group['lr_mult']
			else:
				lr_mult = 1.0
			param_group['lr'] = lr * lr_mult

	"""
	Optimization
	"""
	def fit(self, src_train_iter, tgt_train_iter, optimizer, lr_scheduler, tgt_eval_iter=None, 
			metrics_src=metric.Accuracy(topk=1), metrics_tgt=metric.Accuracy(topk=1), epoch_start=0, epoch_end=1000, **kwargs):

		"""
		checking
		"""
		if kwargs:
			logging.warning("Unknown kwargs: {}".format(kwargs))

		assert torch.cuda.is_available(), "only support GPU version"

		"""
		start the main loop
		"""
		class_weight = torch.from_numpy(np.array([1.0] * self.num_classes)).float()
		if torch.cuda.is_available():
			class_weight = class_weight.cuda()
		pause_sec = 0.
		step_count = 0
		for i_epoch in range(epoch_start, epoch_end):
			self.callback_kwargs['epoch'] = i_epoch
			epoch_start_time = time.time()

			###########
			# 1] TRAINING
			###########
			metrics_tgt.reset()
			metrics_src.reset()
			self.net.train()
			sum_sample_inst_tgt = 0
			sum_sample_inst_src = 0
			sum_sample_elapse = 0.
			sum_update_elapse = 0
			softmax_param = 10.0 # in original PADA, some softmax_param are set to 10.0
			tgt_only = False
			batch_start_time = time.time()
			logging.info("Start epoch {:d}:".format(i_epoch))

			if src_train_iter.__len__() == tgt_train_iter.__len__():
				tgt_only = True

			if src_train_iter.__len__() > tgt_train_iter.__len__():
				train_zip = zip(src_train_iter, cycle(tgt_train_iter))
			elif src_train_iter.__len__() < tgt_train_iter.__len__() and self.da_method is not None:
				train_zip = zip(cycle(src_train_iter), tgt_train_iter)
			else:
				train_zip = zip(src_train_iter, tgt_train_iter)
			# train_zip = zip(src_train_iter, tgt_train_iter)
			train_iter = enumerate(train_zip)

			if tgt_eval_iter is not None:
				update_iter = enumerate(tgt_eval_iter)
				eval_iter = enumerate(tgt_eval_iter)
			else:
				eval_iter = None

			for i_batch, ((src_data, src_label), (tgt_data, tgt_label)) in train_iter:
				self.callback_kwargs['batch'] = i_batch
				step_count += 1

				update_start_time = time.time()

				# [forward] making next step
				if not tgt_data.shape[0] == src_data.shape[0]:
					logging.info("Training:: Target data not match size with source data, with {} target data + {} source data.".format(tgt_data.shape[0], src_data.shape[0]))
					continue
				outputs_tgt, outputs_src, losses, each_losses = self.forward(src_data, tgt_data, src_label, tgt_label, class_weight)

				# [backward]
				optimizer.zero_grad()
				for loss in losses: loss.backward()
				self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.update())
				optimizer.step()

				# [evaluation] update train metric
				metrics_tgt.update([output.data.cpu() for output in outputs_tgt], tgt_label.cpu(), [loss.data.cpu() for loss in losses])
				metrics_src.update([output.data.cpu() for output in outputs_src], src_label.cpu(), [loss.data.cpu() for loss in losses])

				# timing each batch
				sum_sample_elapse += time.time() - batch_start_time
				sum_update_elapse += time.time() - update_start_time
				batch_start_time = time.time()
				sum_sample_inst_tgt += tgt_data.shape[0]
				sum_sample_inst_src += src_data.shape[0]

				if (i_batch % self.step_callback_freq) == 0:
					# retrive eval results and reset metic
					self.callback_kwargs['namevals'] = metrics_src.get_name_value()
					metrics_src.reset()
					# speed monitor
					self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst_src
					self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst_src
					# callbacks
					self.step_end_callback()
					if not tgt_only:
						# retrive eval results and reset metic
						self.callback_kwargs['namevals'] = metrics_tgt.get_name_value()
						metrics_tgt.reset()
						# speed monitor
						self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst_tgt
						self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst_tgt
						# callbacks
						self.step_end_callback()
					sum_update_elapse = 0
					sum_sample_elapse = 0
					sum_sample_inst_tgt = 0
					sum_sample_inst_src = 0

					if each_losses is not None and not (len(each_losses) == 0):
						logging.info("The individual losses are: {}".format(each_losses))

				# torch.cuda.empty_cache()

				# update class weights with prediction (do not use metric auto update here for convenience) (only for PADA)
				if (step_count % self.step_class_w_update) == 0 and self.da_method == 'PADA':
					logging.info("Updating class weight for PADA")
					start_update_proc = True
					self.net.eval()
					assert update_iter is not None
					for j_batch, (tgt_data_update, tgt_label_update) in update_iter:
						preds_up, _, _, _ = self.forward(tgt_data_update, tgt_data_update, tgt_label_update, tgt_label_update, class_weight)
						preds_up = preds_up[0]
						preds_up = nn.Softmax(dim=1)(softmax_param * preds_up)
						if start_update_proc:
							all_preds_up = preds_up.data.cpu().float()
							start_update_proc = False
						else:
							all_preds_up = torch.cat((all_preds_up, preds_up.data.cpu().float()), 0)
					class_weight = torch.mean(all_preds_up, 0)
					class_weight = (class_weight / torch.mean(class_weight)).cuda().view(-1)
					logging.info("Update class weight for PADA complete")
					self.net.train()

			###########
			# 2] END OF EPOCH
			###########
			self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
			self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
			self.epoch_end_callback()

			###########
			# 3] Evaluation
			###########
			if (eval_iter is not None) and ((i_epoch+1) % max(1, int(self.save_freq/2))) == 0:

				logging.info("Start evaluating epoch {:d}:".format(i_epoch))
				# torch.cuda.empty_cache()
				metrics_tgt.reset()
				self.net.eval()
				sum_sample_elapse = 0.
				sum_sample_inst_tgt = 0
				sum_forward_elapse = 0.
				batch_start_time = time.time()

				for i_batch_eval, (tgt_data_eval, tgt_label_eval) in eval_iter:
					self.callback_kwargs['batch'] = i_batch_eval

					forward_start_time = time.time()

					outputs_tgt_eval, _, losses_eval, _ = self.forward(tgt_data_eval, tgt_data_eval, tgt_label_eval, tgt_label_eval, class_weight)

					metrics_tgt.update([output.data.cpu() for output in outputs_tgt_eval], tgt_label_eval.cpu(), [loss.data.cpu() for loss in losses_eval])

					sum_forward_elapse += time.time() - forward_start_time
					sum_sample_elapse += time.time() - batch_start_time
					batch_start_time = time.time()
					sum_sample_inst_tgt += tgt_data_eval.shape[0]

				# evaluation callbacks
				self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst_tgt
				self.callback_kwargs['update_elapse'] = sum_forward_elapse / sum_sample_inst_tgt
				self.callback_kwargs['namevals'] = metrics_tgt.get_name_value()
				self.step_end_callback()

		logging.info("Optimization done!")
