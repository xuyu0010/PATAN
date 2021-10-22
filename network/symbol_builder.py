import logging

from .PATAN import PATAN
from .config import get_config

def get_symbol(name, print_net=False, da_method=None, segments=3, consensus_type='avg', dynamic_reverse=False, start_clip_weight=0, batch_size=16, use_spatial=False, **kwargs):

	logging.info("Network:: Getting symbol with {} domain adaptation using {} network.".format(da_method, name))

	if name.upper() == 'PATAN':
		logging.info("Network:: For frame-based method using {} segments".format(segments))
		net = PATAN(da_method=da_method, segments=segments, consensus_type=consensus_type, dynamic_reverse=dynamic_reverse, 
					start_clip_weight=start_clip_weight, batch_size=batch_size, use_spatial=use_spatial, **kwargs)
	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf
