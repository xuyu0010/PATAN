import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter

def get_hmdb51(data_root='./dataset/HMDB51',
			   clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
			   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for hmdb51
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	# train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_train_uh.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  shuffle_list_seed=(seed+2),
					  )

	# val_sampler = sampler.SequentialSampling(num=clip_length, interval=val_interval, fix_cursor=True, shuffle=True)
	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_test_uh.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  )

	return (train, val)


def get_arid(data_root='./dataset/ARID',
			   clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
			   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for arid
	"""
	logging.debug("VideoIter:: clip_length = {}, segments = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, segments, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	# train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'arid_train_ha.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  shuffle_list_seed=(seed+2),
					  )

	# val_sampler = sampler.SequentialSampling(num=clip_length, interval=val_interval, fix_cursor=True, shuffle=True)
	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'arid_test_ha.txt'),
					  # txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'arid_test_ha.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  )

	return (train, val)


def get_ucf101(data_root='./dataset/UCF101',
			   clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
			   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for ucf-101
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	# train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ucf101_train_uh.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  shuffle_list_seed=(seed+2),
					  )

	# val_sampler = sampler.SequentialSampling(num=clip_length, interval=val_interval, fix_cursor=True, shuffle=True)
	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ucf101_test_hu.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  )

	return (train, val)


def get_minikinetics(data_root='./dataset/MK200',
				 clip_length=8, segments=3, frame_per_seg=1, train_interval=2, val_interval=2,
				 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
				 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, **kwargs):
	""" data iter for mini-kinetics-200
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	# train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'train'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'mk200_train_um.txt'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[256, 320]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  shuffle_list_seed=(seed+2),
					  )

	# val_sampler   = sampler.SequentialSampling(num=clip_length, interval=val_interval, fix_cursor=True, shuffle=True)
	val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, frame_per_seg=frame_per_seg, interval=val_interval, fix_cursor=False, shuffle=True)
	val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'mk200_test_um.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256, 256)),
										 # transforms.RandomScale(make_square=False, aspect_ratio=[1.0, 1.0], slen=[256, 256]),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  )
	return (train, val)


def creat(name, batch_size, num_workers=4, **kwargs):

	if name.upper() == 'HMDB51':
		train, val = get_hmdb51(**kwargs)
	elif name.upper() == 'ARID':
		train, val = get_arid(**kwargs)
	elif name.upper() == 'UCF101':
		train, val = get_ucf101(**kwargs)
	elif name.upper() == 'MK200':
		train, val = get_minikinetics(**kwargs)
	else:
		assert NotImplementedError("iter {} not found".format(name))


	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
	val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

	return (train_loader, val_loader)
