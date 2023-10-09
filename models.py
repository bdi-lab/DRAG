import torch
import torch.nn as nn
from layers import DRAGConv

class DRAG(nn.Module):
	def __init__(self, feature_dim: int, emb_dim: list, gat_heads: list, num_agg_heads: int,
		num_classes: int=2, is_concat: bool=True, num_relations: int=1, feat_drop: float=0.0, attn_drop: float=0.0):
		super(DRAG, self).__init__()
		assert (emb_dim[-1] % num_agg_heads) == 0
		self.num_layers = len(emb_dim)
		self.emb_dim = emb_dim
		self.n_hidden = emb_dim[-1] // num_agg_heads
		self.gat_heads = gat_heads
		self.num_agg_heads = num_agg_heads
		self.num_relations = num_relations
		self.multi_relation = num_relations != 1 
		self.is_concat = is_concat
		
		# Stack DRAGConv layers.
		self.layers = nn.ModuleList()
		for i in range(self.num_layers):
			layers = nn.ModuleList()
			for _ in range(num_relations):
				in_dim = feature_dim if i == 0 else self.emb_dim[i-1]
				out_dim = self.emb_dim[i] // self.gat_heads[i] if self.is_concat else self.emb_dim[i]
				layers.append(DRAGConv(in_dim, out_dim, self.gat_heads[i], feat_drop, attn_drop))
			self.layers.append(layers)
			
		
		# Stack feature projection layers.
		self.feat_layers = nn.ModuleList()
		for i in range(self.num_layers):
			in_dim = self.emb_dim[i-1] if i > 0 else feature_dim
			out_dim = self.emb_dim[i]
			self.feat_layers.append(nn.Linear(in_dim, out_dim))
		
		# Define attention matrices.
		self.rel_attn_Wl_l = nn.ModuleList()
		self.rel_attn_Wr_l = nn.ModuleList()
		self.rel_attn_P_l = nn.ModuleList()
		self.rel_attn_layers = nn.ModuleList()
		for i in range(self.num_layers):
			in_dim = self.emb_dim[i-1] if i > 0 else feature_dim
			out_dim = self.emb_dim[i]
			self.rel_attn_Wl_l.append(nn.Linear(in_dim, out_dim))
			self.rel_attn_Wr_l.append(nn.Linear(out_dim, out_dim))
			self.rel_attn_P_l.append(nn.Linear(out_dim, out_dim))
			self.rel_attn_layers.append(nn.Linear(self.n_hidden, 1))
		self.proj_layer = nn.Linear(feature_dim, self.emb_dim[-1])
		self.hop_attn_Wl = nn.Linear(feature_dim, self.emb_dim[-1])
		self.hop_attn_Wr = nn.Linear(self.emb_dim[-1], self.emb_dim[-1])
		self.hop_attn_P = nn.Linear(self.emb_dim[-1], self.emb_dim[-1])
		self.hop_attn_layer = nn.Linear(self.n_hidden, 1)
		
		# Define projection layers and functions.
		self.linear_layer = nn.Linear(self.emb_dim[-1], num_classes)
		self.activation = nn.LeakyReLU()
		self.softmax = nn.Softmax(dim=1)
	
	def forward(self, blocks, attn_coeff=False):
		edge_types = blocks[0].etypes
		num_nodes = torch.LongTensor([blocks[-1].dstdata["x"].shape[0]])
		x = blocks[0].srcdata['x']
		
		beta_l = []
		layer_features = [self.proj_layer(blocks[-1].dstdata['x'])]
		for i in range(self.num_layers):
			features = []
			for j in range(self.num_relations):
				# eqn (1) & (2) in Node Representation per Relation and Self-Transformation section.
				h = self.layers[i][j](blocks[i][edge_types[j]], x)
				if self.is_concat:
					h = h.reshape(-1, self.emb_dim[i])
				else:
					h = h.mean(dim=1).squeeze()
				features.append(self.activation(h))
			
			# eqn (3) & (4) in Relation-Attentive Aggregation section.
			x_dst = x[:blocks[i].dstdata['x'].shape[0]]
			features.append(self.activation(self.feat_layers[i](x_dst)))
			x = torch.stack(features, 1).squeeze(2)
			if attn_coeff:
				beta, x = self.rel_attn(x, x_dst, layer_num=i, attn_coeff=attn_coeff)
				beta_l.append(beta[:num_nodes])
			else:
				x = self.rel_attn(x, x_dst, layer_num=i, attn_coeff=attn_coeff) 
				
			x = self.activation(x)
			layer_features.append(x[:num_nodes])
		
		# eqn (5) & (6) in Aggregation with Multiple Layers section.
		x = torch.stack(layer_features, 1).squeeze(2)
		if attn_coeff:
			gamma, x = self.hop_attn(x, blocks[-1].dstdata['x'], attn_coeff=attn_coeff)
		else:
			x = self.hop_attn(x, blocks[-1].dstdata['x'], attn_coeff=attn_coeff)
		
		x = self.activation(x)
		out = self.linear_layer(x)
		
		if attn_coeff:
			return out, [beta_l, gamma[:num_nodes]]
		return out
	
	# The attention coefficient beta.
	def rel_attn(self, features, features_prev, layer_num, attn_coeff=False):
		g_l = self.rel_attn_Wl_l[layer_num](features_prev).view(-1, self.num_agg_heads, self.n_hidden)
		g_r = self.rel_attn_Wr_l[layer_num](features).view(-1, self.num_relations+1, self.num_agg_heads, self.n_hidden)
		x = g_l.unsqueeze(dim=1).repeat_interleave(self.num_relations+1, dim=1) + g_r
		e = self.rel_attn_layers[layer_num](self.activation(x)).squeeze(-1)
		e = e - e.amax(dim=1).unsqueeze(1)
		a = self.softmax(e)
		v = self.rel_attn_P_l[layer_num](features).view(-1, self.num_relations+1, self.num_agg_heads, self.n_hidden)

		"""
		- a: [N, 4, 8] = [n, i, j]
		- v: [N, 4, 8, 8] = [n, i, j, k]
		- Linear combination results: [N, 64] = [n, j]
		"""
		x = torch.einsum('nij,nijk->njk', a, v).view(-1, self.emb_dim[-1])
		if attn_coeff:
			return a, x
		return x
	
	# The attention coefficient gamma.
	def hop_attn(self, features, features_prev, attn_coeff=False):
		g_l = self.hop_attn_Wl(features_prev).view(-1, self.num_agg_heads, self.n_hidden)
		g_r = self.hop_attn_Wr(features).view(-1, self.num_layers+1, self.num_agg_heads, self.n_hidden)
		x = g_l.unsqueeze(dim=1).repeat_interleave(self.num_layers+1, dim=1) + g_r
		e = self.hop_attn_layer(self.activation(x)).squeeze(-1)
		e = e - e.amax(dim=1).unsqueeze(1)
		a = self.softmax(e)
		v = self.hop_attn_P(features).view(-1, self.num_layers+1, self.num_agg_heads, self.n_hidden)
		
		"""
		- a: [N, 4, 8] = [n, i, j]
		- v: [N, 4, 8, 8] = [n, i, j, k]
		- Linear combination results: [N, 64] = [n, j]
		"""
		x = torch.einsum('nij,nijk->njk', a, v).view(-1, self.emb_dim[-1])
		if attn_coeff:
			return a, x
		return x

	# Compute model prediction with batches.
	def to_prob(self, blocks):
                scores = torch.softmax(self.forward(blocks), dim=1)
                return scores