import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvalMetric(object):

	def __init__(self, name, **kwargs):
		self.name = str(name)
		self.reset()

	def update(self, preds, labels, losses):
		raise NotImplementedError()

	def reset(self):
		self.num_inst = 0
		self.sum_metric = 0.0

	def get(self):
		if self.num_inst == 0:
			return (self.name, float('nan'))
		else:
			return (self.name, self.sum_metric / self.num_inst)

	def get_name_value(self):
		name, value = self.get()
		if not isinstance(name, list):
			name = [name]
		if not isinstance(value, list):
			value = [value]
		return list(zip(name, value))

	def check_label_shapes(self, preds, labels):
		# raise if the shape is inconsistent
		if (type(labels) is list) and (type(preds) is list):
			label_shape, pred_shape = len(labels), len(preds)
		else:
			label_shape, pred_shape = labels.shape[0], preds.shape[0]

		if label_shape != pred_shape:
			raise NotImplementedError("")


class MetricList(EvalMetric):
	"""Handle multiple evaluation metric
	"""
	def __init__(self, *args, name="metric_list"):
		assert all([issubclass(type(x), EvalMetric) for x in args]), \
			"MetricList input is illegal: {}".format(args)
		self.metrics = [metric for metric in args]
		super(MetricList, self).__init__(name=name)

	def update(self, preds, labels, losses=None):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels
		losses = [losses] if type(losses) is not list else losses

		for metric in self.metrics:
			metric.update(preds, labels, losses)

	def reset(self):
		if hasattr(self, 'metrics'):
			for metric in self.metrics:
				metric.reset()
		else:
			logging.warning("No metric defined.")

	def get(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get())
		return ouputs

	def get_name_value(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get_name_value())        
		return ouputs


####################
# COMMON METRICS
####################

class Accuracy(EvalMetric):
	"""Computes accuracy classification score.
	"""
	def __init__(self, name='accuracy', topk=1):
		super(Accuracy, self).__init__(name)
		self.topk = topk

	def update(self, preds, labels, losses):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels

		self.check_label_shapes(preds, labels)
		for pred, label in zip(preds, labels):
			assert self.topk <= pred.shape[1], \
				"topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
			_, pred_topk = pred.topk(self.topk, 1, True, True)

			pred_topk = pred_topk.t()
			correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))

			self.sum_metric += float(correct.view(-1).float().sum(0, keepdim=True).numpy())
			self.num_inst += label.shape[0]


class Loss(EvalMetric):
	"""Dummy metric for directly printing loss.
	"""        
	def __init__(self, name='loss'):
		super(Loss, self).__init__(name)

	def update(self, preds, labels, losses):
		assert losses is not None, "Loss undefined."
		for loss in losses:
			self.sum_metric += float(loss.numpy().sum())
			self.num_inst += loss.numpy().size


def PATA(inputs, adv_outputs, src_label, class_weight):
	weight_ad = torch.zeros(inputs.size(0))
	label_numpy = src_label.data.cpu().numpy()

	for j in range(int(inputs.size(0) / 2)):
		weight_ad[j] = class_weight[int(label_numpy[j])]
	weight_ad = weight_ad / torch.max(weight_ad[0:int(inputs.size(0)/2)])
	for j in range(int(inputs.size(0) / 2), inputs.size(0)):
		weight_ad[j] = 1.0

	batch_size = inputs.size(0) // 2
	dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
	if torch.cuda.is_available():
		dc_target = dc_target.cuda()
		weight_ad = weight_ad.cuda()

	patan_criterion = nn.BCELoss(weight=weight_ad.view(-1))
	if torch.cuda.is_available():
		patan_criterion = patan_criterion.cuda()

	loss = patan_criterion(adv_outputs.view(-1), dc_target.view(-1))
	return loss


def DANN(src_input, tgt_input, adv_outputs):
	dann_criterion = nn.CrossEntropyLoss()
	if torch.cuda.is_available():
		dann_criterion = dann_criterion.cuda()

	src_domain_label = torch.ones(src_input.size(0)).long()
	tgt_domain_label = torch.zeros(tgt_input.size(0)).long()
	domain_label = torch.cat((src_domain_label,tgt_domain_label),0)
	if torch.cuda.is_available():
		domain_label = domain_label.cuda(non_blocking=True)

	loss = dann_criterion(adv_outputs, domain_label)
	return loss
