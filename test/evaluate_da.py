import sys
sys.path.append("..")

import os
import time
import json
import logging
import argparse
import numpy as np
from datetime import date

import torch
import torch.backends.cudnn as cudnn

import dataset
from train.da_model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol
torch.backends.cudnn.enabled = False
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="Video PDA Parser (Evaluation)")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,)
# io
parser.add_argument('--dataset', default='UCF-HMDB')
parser.add_argument('--src-dataset', default='HMDB51')
parser.add_argument('--tgt-dataset', default='UCF101')
parser.add_argument('--frame-interval', type=int, default=2)
parser.add_argument('--task-name', type=str, default='../exps/models/PATAN')
parser.add_argument('--model-dir', type=str, default="./",)
parser.add_argument('--log-file', type=str, default="./test-results/eval-patan")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7", help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='patan')
parser.add_argument('--da-method', default='PATA')
parser.add_argument('--segments', type=int, default=5)
parser.add_argument('--frame_per_seg', type=int, default=1)
parser.add_argument('--consensus_type', type=str, default='trn-m')
parser.add_argument('--start-clip-weight', type=int, default=2000)
# evaluation
parser.add_argument('--load-epoch', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--list-file', type=str, default='ucf101_test_hu.txt',)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--test-rounds', type=int, default=15)


def autofill(args):
	# customized
	if not args.task_name:
		args.task_name = os.path.basename(os.getcwd())
	# fixed
	args.model_prefix = os.path.join(args.model_dir, args.task_name)
	return args


def set_logger(log_file='', debug_mode=False):

	log_id = 1

	if not ".log" in log_file:
		today = date.today()
		today = today.strftime("%m%d")
		log_file = log_file + "-" + today + "-" + str(log_id) + ".log"

	if log_file:
		if not os.path.exists("./"+os.path.dirname(log_file)):
			os.makedirs("./"+os.path.dirname(log_file))
		while os.path.exists(log_file):
			log_id = int(log_file.split('-')[-1].split('.')[0])
			new_log_id = log_id + 1
			k = log_file.rfind(str(log_id))
			log_file = log_file[:k] + str(new_log_id) + log_file[k+1:]

		handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	else:
		handlers = [logging.StreamHandler()]

	""" add '%(filename)s' to format show source file """
	logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers = handlers)


if __name__ == '__main__':

	# set args
	args = parser.parse_args()
	args = autofill(args)

	set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
	logging.info("Start evaluation with args:\n" + json.dumps(vars(args), indent=4, sort_keys=True))

	# set device states
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
	assert torch.cuda.is_available(), "CUDA is not available"

	# load dataset related configuration
	dataset_cfg = dataset.get_config(name=args.dataset, src=args.src_dataset)
	num_classes = dataset_cfg['num_classes']

	# creat model
	sym_net, input_config = get_symbol(name=args.network, da_method=args.da_method, segments=args.segments, consensus_type=args.consensus_type,
										start_clip_weight=args.start_clip_weight, batch_size=args.batch_size, **dataset_cfg)

	# network
	if torch.cuda.is_available():
		sym_net = torch.nn.DataParallel(sym_net).cuda()
	else:
		sym_net = torch.nn.DataParallel(sym_net)
	net = static_model(net=sym_net,da_method=args.da_method,model_prefix=args.model_prefix)
	net.load_checkpoint(epoch=args.load_epoch)

	# data iterator:
	data_root = "../dataset/{}".format(args.tgt_dataset)
	target_mean, target_std = input_config['mean'], input_config['std']
	if args.tgt_dataset == 'ARID':
		target_mean = [0.079612, 0.073888, 0.072454]
		target_std = [0.100459, 0.09705, 0.089911]
	normalize = transforms.Normalize(mean=target_mean, std=target_std)
	val_sampler = sampler.SegmentalSampling(num_per_seg=args.clip_length,segments=args.segments,frame_per_seg=args.frame_per_seg,interval=args.frame_interval,fix_cursor=False,shuffle=True)
	val_loader = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'), # Add 'val' for MK200
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', args.list_file),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										transforms.Resize((256,256)),
										transforms.RandomCrop((224,224)),
										transforms.ToTensor(),
										normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )
	eval_iter = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

	# eval metrics
	metrics = metric.MetricList(metric.Loss(name="loss-ce"), metric.Accuracy(topk=1, name="top1"), metric.Accuracy(topk=5, name="top5"))
	metrics.reset()

	# main loop
	net.net.eval()
	avg_score = {}
	sum_batch_elapse = 0.
	sum_batch_inst = 0
	duplication = 1
	step = 100000
	softmax = torch.nn.Softmax(dim=1)
	class_weight = torch.from_numpy(np.array([1.0] * num_classes)).float()
	if torch.cuda.is_available():
		class_weight = class_weight.cuda()

	total_round = args.test_rounds
	for i_round in range(total_round):
		i_batch = 0
		logging.info("round #{}/{}".format(i_round, total_round))
		
		for data, target, video_subpath in eval_iter:
			
			batch_start_time = time.time()

			if 'PATAN' in net.net.module.__class__.__name__:
				outputs, _, _, _, _, losses, _, _ = net.forward(data, data, target, target, class_weight, step)
			else:
				outputs, _, losses, _, _ = net.forward(data, data, target, target, class_weight, step)

			sum_batch_elapse += time.time() - batch_start_time
			sum_batch_inst += 1

			# recording
			output = softmax(outputs[0]).data.cpu()
			target = target.cpu()
			losses = losses[0].data.cpu()
			for i_item in range(0, output.shape[0]):
				output_i = output[i_item,:].view(1, -1)
				target_i = torch.LongTensor([target[i_item]])
				loss_i = losses
				video_subpath_i = video_subpath[i_item]
				if video_subpath_i in avg_score:
					avg_score[video_subpath_i][2] += output_i
					avg_score[video_subpath_i][3] += 1
					duplication = 0.92 * duplication + 0.08 * avg_score[video_subpath_i][3]
				else:
					avg_score[video_subpath_i]=[torch.LongTensor(target_i.numpy().copy()),torch.FloatTensor(loss_i.numpy().copy()),torch.FloatTensor(output_i.numpy().copy()),1]

			# show progress
			if (i_batch % 100) == 0:
				metrics.reset()
				for _, video_info in avg_score.items():
					target, loss, pred, _ = video_info
					metrics.update([pred], target, [loss])
				name_value = metrics.get_name_value()
				logging.info("{:.1f}%, {:.1f} \t| Batch [0,{}]    \tAvg: {} = {:.5f}, {} = {:.5f}, {} = {:.5f}".format(
							float(100*i_batch)/eval_iter.__len__(), duplication, i_batch, \
							name_value[0][0][0], name_value[0][0][1], \
							name_value[1][0][0], name_value[1][0][1], \
							name_value[2][0][0], name_value[2][0][1]))
			i_batch += 1


	# finished
	logging.info("Evaluation Finished!")

	metrics.reset()
	for _, video_info in avg_score.items():
		target, loss, pred, _ = video_info
		metrics.update([pred], target, [loss])

	logging.info("Total time cost: {:.1f} sec".format(sum_batch_elapse))
	logging.info("Speed: {:.4f} samples/sec".format(args.batch_size * sum_batch_inst / sum_batch_elapse))
	logging.info("Accuracy:")
	logging.info(json.dumps(metrics.get_name_value(), indent=4, sort_keys=True))
