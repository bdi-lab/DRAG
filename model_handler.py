import os
import time
import random
import argparse
from typing import Tuple

import torch
import numpy as np

from models import *
from datasets import *
from data_handler import DataHandlerModule
from result_manager import ResultManager
from utils import test, generate_batch_idx

PYG_DIR_PATH = "./data/pyg"

class ModelHandlerModule():
	def __init__(self, configuration, datahandler: DataHandlerModule):
		self.args = argparse.Namespace(**configuration)
		self.dataset = datahandler.dataset
		self.epochs = self.args.epochs
		self.patience = self.args.patience
		self.result = ResultManager(args=configuration)
		
		# Set the seeds and CUDA ID.
		self.seed = self.args.seed
		device = torch.device(self.args.cuda_id)
		torch.cuda.set_device(device)
		
		# Select the model according to the json configuration file.
		self.model = self.select_model()
		self.model.cuda()
		
	def set_seed(self) -> None:
		"""
  		Set the seed for reproducibility. 
    		"""
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		torch.cuda.manual_seed(self.seed)
		torch.cuda.manual_seed_all(self.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		os.environ['PYTHONHASHSEED'] = str(self.seed) 
		
	def select_model(self) -> nn.Module:
		"""
		Select the model according to the configuration.
		If you have imported additional models, you can use it by adding the bellow codes.

		- If the GNN model is homogeneous, adjacemcy matrix is equal to the homogeneous graph.
		- It is determined in the datahandler.
		"""
		torch.cuda.empty_cache()
		graph = self.dataset['graph']
		feature = self.dataset['features']

		model = DRAG(feature.shape[1], self.args.emb_size, gat_heads=self.args.n_head,
               num_agg_heads=self.args.n_head_agg, num_classes=2, is_concat=True,
               num_relations=len(graph.etypes), feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop) 
		
		return model
	
	def train(self) -> Tuple[np.array, np.array, float]:
		"""
		Train DRAG model on the fraud detection dataset.
		It returns
			- Prediction: Pseudo labels of total nodes (n,).
			- Confidence: Confidence score for the prediction of the model (n,).
			- Test AUC-ROC : AUC-ROC of the test set. 
		"""
		
		# [STEP-1] Set the seed for various libraries and CUDA ID.
		self.set_seed()
		torch.cuda.empty_cache()
		device = torch.device(self.args.cuda_id)
		torch.cuda.set_device(device)
		
		# [STEP-2] Define the node indices of train/valid/test process and labels.
		graph = self.dataset['graph']
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
		
		# [STEP-3-1] Define the model and loss function.
		model = self.model
		loss_fn = nn.CrossEntropyLoss()
		
		# [STEP-3-2] Define the batch sampler.
		sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.args.emb_size))
  
  		# [STEP-3-3] Initialize the optimizer with learning rate and weight decay rate.
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)
		
  		# [STEP-3-4] Initialize the performance evaluation measure for validation.
		auc_best, f1_mac_best, epoch_best = 1e-10, 1e-10, 0
		
		# [STEP-4] Train the model.
		print("\n", "*"*20, f" Train the DRAG ", "*"*20)
		for epoch in range(self.epochs):
			model.train()
			avg_loss = []
			loss, epoch_time = 0.0, 0.0
			torch.cuda.empty_cache()

			# [STEP-4-1] Generate batch indices for train loader.
			batch_idx = generate_batch_idx(idx_train, y_train, self.args.batch_size, self.args.seed)
			train_loader = dgl.dataloading.DataLoader(graph, batch_idx, sampler, batch_size=self.args.batch_size, shuffle=False, drop_last=False, use_uva=True)
			
			start_time = time.time()
			for batch in train_loader:
				# [STEP-4-2] Set the batche nodes.
				_, output_nodes, blocks = batch
				blocks = [b.to(device) for b in blocks]
				output_labels = blocks[-1].dstdata['y'].type(torch.LongTensor).cuda()
				
				# [STEP-4=3] Compute the loss of the model.
				"""
				 - If loss is nn.CrossEntropyLoss, the data type of the labels should be LongTensor.
				 - If loss is nn.BCELoss, the data type of the labels should be FloatTensor.
				"""
				logits = model(blocks)
				loss = loss_fn(logits, output_labels.squeeze())
				
				# [STEP-4-4] Compute the gradient and update the model.
				loss.backward()
				optimizer.step()

				# [STEP-4-5] Clear the remain gradient as zero value.
				optimizer.zero_grad()
				avg_loss.append(loss.item() / output_nodes.shape[0]) # Calculate average train loss.
			
			# [STEP-4-6] Write the train log.
			end_time = time.time()
			epoch_time += end_time - start_time
			line = f'Epoch: {epoch+1} (Best: {epoch_best}), loss: {np.mean(avg_loss)}, time: {epoch_time}s'
			self.result.write_train_log(line, print_line=True)
			
			# [STEP-5] Validate the model performance for each validation epoch.
			if (epoch+1) % self.args.valid_epochs == 0:
				model.eval()
				# [STEP-5-1] Calculate the AUC, Recall, F1-macro, Precision with validation set.
				auc_val, recall_val, f1_mac_val, precision_val = test(model, self.dataset['valid_loader'], self.result, epoch, epoch_best, flag="val")
				
				# [STEP-5-2] If the current model is best, save the model and update the best value. 
				gain_auc = (auc_val - auc_best)/auc_best
				gain_f1_mac =  (f1_mac_val - f1_mac_best)/f1_mac_best
				if (gain_auc + gain_f1_mac) > 0:
					auc_best, recall_best, f1_mac_best, precision_best, epoch_best = auc_val, recall_val, f1_mac_val, precision_val, epoch
					torch.save(model.state_dict(), self.result.model_path)
			
			# [STEP-6] Test early stopping condition.
			if (epoch - epoch_best) > self.args.patience:
				print("\n", "*"*20, f"Early stopping at epoch {epoch}", "*"*20, )
				break
		# [STEP-7] Write the best validation results for model selection.
		self.result.write_val_log(auc_best, recall_best, f1_mac_best, precision_best, epoch_best)
		
		# [STEP-8] Load the best model with repect to the validation performance.
		print("Restore model from epoch {}".format(epoch_best))
		model.load_state_dict(torch.load(self.result.model_path))
		
		# [STEP-9] Test the model performance.
		print("\n", "*"*20, f" Test the DRAG ", "*"*20)
		auc_test, recall_test, f1_mac_test, precision_test = test(model, self.dataset['test_loader'], self.result, epoch, epoch_best, flag="test")
		
		return auc_test, f1_mac_test