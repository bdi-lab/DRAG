import os
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score

def generate_batch_idx(idx_train, y_train, batch_size, seed):
	nodes, labels = shuffle(idx_train, y_train, random_state=seed)
					
	nodes = np.array(nodes)
	labels = np.array(labels)

	pos_idx = nodes[labels == 1]
	neg_idx = nodes[labels == 0]

	batch_idx = np.empty(0,)
	num_batches = int(labels.sum() * 2 / batch_size) + 1
	for batch in range(num_batches):
		pos_start = int(batch * batch_size / 2)
		pos_end = int(min((batch+1) * batch_size / 2, len(pos_idx)))
		pos = pos_idx[pos_start:pos_end]
		neg = np.random.choice(neg_idx, len(pos), replace=False)
		batch_idx = np.append(batch_idx, pos)
		batch_idx = np.append(batch_idx, neg)
	
	batch_idx = torch.LongTensor(batch_idx)
	return batch_idx

def test(model, loader, result, epoch, epoch_best, flag=None, valid_thresh=None):

	labels = []
	output_list = [[], [], []]
	for input_nodes, output_nodes, blocks in loader:
		output_labels = blocks[-1].dstdata['y'].data.cpu().numpy()
		output = model.to_prob(blocks).data.cpu().numpy()
		
		prediction = output.argmax(axis=1)
		confidence = output.max(axis=1)
		anomaly_confidence = output[:, 1]

		output_list[0].extend(prediction.tolist())
		output_list[1].extend(confidence.tolist())
		output_list[2].extend(anomaly_confidence.tolist())
		labels.extend(output_labels.tolist())
	output_list = np.array(output_list)
	labels = np.array(labels)
	
	f1 = f1_score(labels, output_list[0])
	f1_macro = f1_score(labels, output_list[0], average='macro')
	precision = precision_score(labels, output_list[0], zero_division=0)
	precision_macro = precision_score(labels, output_list[0], zero_division=0, average='macro')
	accuracy = accuracy_score(labels, output_list[0])
	recall = recall_score(labels, output_list[0])
	recall_macro = recall_score(labels, output_list[0], average='macro')
	auc = roc_auc_score(labels, output_list[2])

	line= f"- F1: {f1:.4f}\t- Recall: {recall:.4f}\t- Precision: {precision:.4f}\t- Accuracy: {accuracy:.4f}\t- AUC-ROC: {auc:.4f}\t- F1-macro: {f1_macro:.4f}\t- Recall-macro: {recall_macro:.4f}\t- AP: {precision_macro:.4f}\t\n"

	if flag == "test":
			result.write_test_log(epoch_best, accuracy, f1, f1_macro, precision, precision_macro, recall, recall_macro, auc, line)
	elif flag == "val":
			result.write_train_log("[Validation Performance]")
			result.write_train_log(line)
	return auc, recall, f1_macro, precision

def collect_settings(exp_result_path: str) -> list:
	return list(set(["-".join(filename.split('-')[:3]) for filename in os.listdir(os.path.join(exp_result_path, 'test_log'))]))

def collect_results(exp_result_path: str) -> None:
	settings = collect_settings(exp_result_path)
	
	create_dir(f'{exp_result_path}/validation_df')
	create_dir(f'{exp_result_path}/test_df')
	for setting in settings:
		load_df_val(exp_result_path, setting)
		load_df_test(exp_result_path, setting)
	
def load_df_test(exp_result_path: str, setting :str) -> pd.DataFrame:
	df_test = pd.DataFrame()
	log_test_dir = os.path.join(exp_result_path, 'test_log')
	log_test_path_l = [os.path.join(log_test_dir, filename) for filename in os.listdir(log_test_dir) if setting in filename]
	for path in log_test_path_l:
		with open(path, 'r') as f:
			idx = len(df_test)
			lines = [line.strip() for line in f.readlines()][:-1]
			result = lines.pop()
			if (not 'Test performance' in result):
				continue
			exp_id = path.split("\\")[-1][:-4]
			df_test.loc[idx, 'exp_id'] = exp_id
			result = dict([tuple(metric.strip().split(': ')) for metric in result.split('- ')[1:]])
			result = dict([(k.lower(), float(v)) for (k, v) in result.items()])
			df_test.loc[idx, 'epoch_best'] = int(result['epoch_best'])
			df_test.loc[idx, 'accuracy'] = float(result['accuracy'])
			df_test.loc[idx, 'f1'] = float(result['f1'])
			df_test.loc[idx, 'f1_macro'] = float(result['f1-macro'])
			df_test.loc[idx, 'precision'] = float(result['precision'])
			df_test.loc[idx, 'precision_macro'] = float(result['ap'])
			df_test.loc[idx, 'recall'] = float(result['recall'])
			df_test.loc[idx, 'recall_macro'] = float(result['recall-macro'])
			df_test.loc[idx, 'auc'] = float(result['auc-roc'])
			args = dict([tuple(line.split(': ')) for line in lines])
			for key in sorted(args.keys()):
				df_test.loc[idx, key] = args[key]
	
	df_test_dir = f"{exp_result_path}/test_df"
	df_test_path = os.path.join(df_test_dir, f"{setting}.pkl")
	df_test.to_pickle(df_test_path)
	return df_test

def load_df_val(exp_result_path: str, setting :str) -> pd.DataFrame:
	df_val = pd.DataFrame()
	log_val_dir = os.path.join(exp_result_path, 'validation_log')
	log_val_path_l = [os.path.join(log_val_dir, filename) for filename in os.listdir(log_val_dir) if setting in filename]
	for path in log_val_path_l:
		with open(path, 'r') as f:
			idx = len(df_val)
			lines = [line.strip() for line in f.readlines()][:-1]
			result = lines.pop()
			if (not 'Validation performance' in result):
				continue
			exp_id = path.split("\\")[-1][:-4]
			df_val.loc[idx, 'exp_id'] = exp_id
			result = dict([tuple(metric.strip().split(': ')) for metric in result.split('- ')[1:]])
			result = dict([(k.lower(), float(v)) for (k, v) in result.items()])
			df_val.loc[idx, 'epoch_best'] = int(result['epoch_best'])
			df_val.loc[idx, 'f1_macro'] = float(result['f1-macro'])
			df_val.loc[idx, 'precision'] = float(result['precision'])
			df_val.loc[idx, 'recall'] = float(result['recall'])
			df_val.loc[idx, 'auc'] = float(result['auc-roc'])
			args = dict([tuple(line.split(': ')) for line in lines])
			for key in sorted(args.keys()):
				df_val.loc[idx, key] = args[key]
				
	df_val_dir = f"{exp_result_path}/validation_df"
	df_val_path = os.path.join(df_val_dir, f"{setting}.pkl")            
	df_val.to_pickle(df_val_path)
	return df_val

def get_df_val_paths(exp_result_path: str, dataset: str, train_ratio: str):
	dir_path = f"{exp_result_path}/validation_df"
	setting = '-'.join([dataset, train_ratio])
	df_val_l = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if (setting in filename)]
	if df_val_l == 0:
		print(f"There is no result for {setting} in {exp_result_path}")
		return None
	return df_val_l

def sort_and_find(df_path: str):
	df = pd.read_pickle(df_path)
	df["mean"] = df[["auc", "f1_macro"]].mean(axis=1)
	df.sort_values(by='mean', ascending=False, inplace=True)
	return df.iloc[0]['exp_id']

def get_best_exp_id(exp_result_path: str, dataset: str, train_ratio: str, seed: str):
	dir_path = f"{exp_result_path}/validation_df"
	setting = '-'.join([dataset, train_ratio, seed])
	df_val_path = os.path.join(dir_path, f"{setting}.pkl")
	if not os.path.exists(df_val_path):
		print(f"There is no result for {setting} in {exp_result_path}")
		return None
	return sort_and_find(df_val_path)

def get_best_exp_ids(exp_result_path: str, dataset: str, train_ratio: str):
	df_val_l = get_df_val_paths(exp_result_path, dataset, train_ratio)
	if df_val_l == None:
		return None
	
	exp_ids = []
	for df_val_path in df_val_l:
		exp_ids.append(sort_and_find(df_val_path))
	return exp_ids

def get_test_result(exp_result_path: str, exp_id: str):
	dir_path = f"{exp_result_path}/test_df"
	setting = '-'.join(exp_id.split('-')[:3])
	df_test_path = os.path.join(dir_path, f"{setting}.pkl")
	df_test = pd.read_pickle(df_test_path)
	return df_test[df_test['exp_id']==exp_id][['exp_id', 'f1_macro', 'auc', 'seed', 'train_ratio']]

def get_test_results(exp_result_path: str, exp_ids: list):
	dir_path = f"{exp_result_path}/test_df"
	settings = ['-'.join(exp_id.split('-')[:3]) for exp_id in exp_ids]
	
	df_result = pd.DataFrame()
	for setting in settings:
		df_test_path = os.path.join(dir_path, f"{setting}.pkl")
		df_test = pd.read_pickle(df_test_path)
		df_temp = df_test[df_test['exp_id'].isin(exp_ids)][['exp_id', 'f1_macro', 'auc', 'seed', 'train_ratio']]
		df_result = pd.concat([df_result, df_temp], ignore_index=True)
	return df_result

def create_dir(dir_path) -> None:
	try:
		if not os.path.exists(dir_path):
			os.makedirs(dir_path, exist_ok=True)
	except OSError:
		print("Error: Failed to create the directory.")