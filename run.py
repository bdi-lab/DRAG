import argparse, json

from model_handler import *
from data_handler import *
from datasets import *
from utils import *
from models import *

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_config_path', type=str, default='./template.json')
	args = vars(parser.parse_args())
	return args

def main(args) -> None:
	# [STEP-1] Open the configuration file and get arguments.
	with open(args['exp_config_path']) as f:
		args = json.load(f)

	# [STEP-2] Initialize the Datahandler object to handle the fraud detection dataset.
	data_handler = DataHandlerModule(args)

	# [STEP-3] Train model and evaluate the performance.
	model = ModelHandlerModule(args, data_handler)
	_, _ = model.train()

if __name__ == '__main__':
	args = get_arguments()
	main(args)