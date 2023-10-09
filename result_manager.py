import os
from datetime import datetime
from utils import create_dir

class ResultManager:
	"""
	ResultManager manages and saves results of model training and testing based on args.
		- Train log txt file is saved in self.log_train_dir.
		- Validation log txt file is saved in self.log_val_dir.
		- Test log txt file is saved in self.log_test_dir.
		- Model checkpoint pickle file is saved in self.saved_model_dir.
	"""
	def __init__(self, args) -> None:
		self.exp_res_dir = args['save_dir']
		self.log_train_dir = f"{self.exp_res_dir}/train_log"
		self.log_val_dir = f"{self.exp_res_dir}/validation_log"
		self.log_test_dir = f"{self.exp_res_dir}/test_log"
		self.saved_model_dir = f"{self.exp_res_dir}/saved_models"
		
		create_dir(self.exp_res_dir)
		create_dir(self.saved_model_dir)
		create_dir(self.log_train_dir)
		create_dir(self.log_val_dir)
		create_dir(self.log_test_dir)
		
		self.args = args
		data_name = args['data_name']
		self.setting_name = f"{data_name}-{str(int(args['train_ratio']*100)).zfill(2)}-{str(args['seed']).zfill(3)}"
		self.exp_id = f"{self.setting_name}-{datetime.now().strftime('%y%m%d-%H%M%S-%f')}"
		self.log_train_path = os.path.join(self.log_train_dir, f"{self.exp_id}.log")
		self.log_val_path = os.path.join(self.log_val_dir, f"{self.exp_id}.log")
		self.log_test_path = os.path.join(self.log_test_dir, f"{self.exp_id}.log")
		self.model_path = os.path.join(self.saved_model_dir, f"{self.exp_id}.pickle")
		
		self.init_logs()
			
	def get_configuration_line(self) -> str:
		line = ""
		for key in sorted(self.args.keys()):
			line = f"{line}\n{key}: {self.args[key]}"
		return line
	
	def init_logs(self):
		line = self.get_configuration_line()[1:]
		with open(self.log_train_path, 'a') as file:
			file.write(line + "\n")
		with open(self.log_val_path, 'a') as file:
			file.write(line + "\n")
		with open(self.log_test_path, 'a') as file:
			file.write(line + "\n")
	
	def write_train_log(self, line, print_line=False):
		with open(self.log_train_path, 'a') as file:
			file.write(line + "\n")
			if print_line: print(line)
	
	def write_val_log(self, auc_best: float, recall_best: float,
					  f1_mac_best: float, precision_best: float, epoch_best: int) -> None:
		
		with open(self.log_val_path, 'a') as file:
			line_eval= f"- AUC-ROC: {auc_best:.4f}\t- Recall: {recall_best:.4f}\t- F1-macro: {f1_mac_best:.4f}\t- Precision: {precision_best:.4f}\n"
			line = f"Validation performance: - Epoch_Best: {epoch_best}\t{line_eval}"
			file.write(line + "\n")
			print(line)
	
	def write_test_log(self, epoch_best: int, accuracy: float,
					   f1: float, f1_macro: float,
					   precision: float, precision_macro: float,
					   recall: float, recall_macro: float, auc: float, line: str) -> None:
		
		with open(self.log_test_path, 'a') as file:
			line = f"Test performance: - Epoch_Best: {epoch_best}\t{line}"
			file.write(line + "\n")
			print(line)
