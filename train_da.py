import os
import json
import socket
import logging
import argparse
from datetime import date

import torch
import torch.nn.parallel
import torch.distributed as dist

import dataset
from train_da_model import train_da_model
from network.symbol_builder import get_symbol
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="Video PDA Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True, help="print setting for debugging.")
# io
parser.add_argument('--dataset', default='UCF-HMDB')
parser.add_argument('--src-dataset', default='HMDB51')
parser.add_argument('--tgt-dataset', default='UCF101')
parser.add_argument('--train-frame-interval', type=int, default=2)
parser.add_argument('--val-frame-interval', type=int, default=2)
parser.add_argument('--task-name', type=str, default='')
parser.add_argument('--model-dir', type=str, default="./exps/models")
parser.add_argument('--log-file', type=str, default="")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7")
# algorithm
parser.add_argument('--network', type=str, default='PATAN')
parser.add_argument('--da-method', type=str, default='PATA')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--resume-epoch', type=int, default=-1, help="resume train")
parser.add_argument('--segments', type=int, default=5)
parser.add_argument('--frame_per_seg', type=int, default=1, help="frames sampled per segment")
parser.add_argument('--consensus_type', type=str, default='trn-m', help="tsn consensus type")
parser.add_argument('--dynamic_reverse', action='store_true', help="use dynamic GRL")
parser.add_argument('--start-clip-weight', type=int, default=600, help="steps to start class weight update")
parser.add_argument('--use-spatial', action='store_true')
# optimization
parser.add_argument('--fine-tune', type=bool, default=True)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr-base', type=float, default=0.0025)
parser.add_argument('--lr-steps', type=list, default=[int(1e4*x) for x in [1.5,3,4]])
parser.add_argument('--lr-factor', type=float, default=0.1)
parser.add_argument('--save-freq', type=float, default=10)
parser.add_argument('--step-class-w-update', type=int, default=600, help="steps per class weight update")
parser.add_argument('--end-epoch', type=int, default=50)
parser.add_argument('--random-seed', type=int, default=1)

def autofill(args):
	# customized
	today = date.today()
	today = today.strftime("%m%d")
	if not args.task_name:
		args.task_name = os.path.basename(os.getcwd())
	if not args.log_file:
		if os.path.exists("./exps/logs"):
			args.log_file = "./exps/logs/{}-{}_at-{}.log".format(args.task_name, today, socket.gethostname())
		else:
			args.log_file = ".{}-{}_at-{}.log".format(args.task_name, today, socket.gethostname())
	# fixed
	args.model_prefix = os.path.join(args.model_dir, args.task_name)
	return args

def set_logger(log_file='', debug_mode=False):
	if log_file:
		if not os.path.exists("./"+os.path.dirname(log_file)):
			os.makedirs("./"+os.path.dirname(log_file))
		handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	else:
		handlers = [logging.StreamHandler()]

	logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers = handlers)

if __name__ == "__main__":

	# set args
	args = parser.parse_args()
	args = autofill(args)

	set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
	logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))

	logging.info("Start training with args:\n" + json.dumps(vars(args), indent=4, sort_keys=True))

	# set device states
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
	assert torch.cuda.is_available(), "CUDA is not available"
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)

	# load dataset related configuration
	dataset_cfg = dataset.get_config(name=args.dataset, src=args.src_dataset)

	net, input_conf = get_symbol(name=args.network, pretrained=args.pretrained if args.resume_epoch<0 else None, 
						da_method=args.da_method, segments=args.segments, consensus_type=args.consensus_type, 
						dynamic_reverse=args.dynamic_reverse, start_clip_weight=args.start_clip_weight, use_spatial=args.use_spatial,
						batch_size=args.batch_size, **dataset_cfg)

	# training
	kwargs = {}
	kwargs.update(dataset_cfg)
	kwargs.update({'input_conf': input_conf})
	kwargs.update(vars(args))
	train_da_model(sym_net=net, **kwargs)
