import logging

def get_config(name, src):

	config = {}

	if name.upper() == 'HMDB-ARID':
		config['num_classes'] = 10
	elif name.upper() == 'UCF-HMDB':
		config['num_classes'] = 14
	elif name.upper() == 'MK200-UCF':
		config['num_classes'] = 45
	else:
		logging.error("Configs for dataset '{}' with source dataset {} not found".format(name, src))
		raise NotImplemented

	logging.debug("Dataset: '{}' with src '{}', configs: {}".format(name.upper(), src.upper(), config))

	return config


if __name__ == "__main__":
	logging.getLogger().setLevel(logging.DEBUG)

	logging.info(get_config("HMDB-ARID", "HMDB51"))
