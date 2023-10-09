"""DRAGConv is based on the torch module for graph attention networks v2 (GATv2)."""

import torch as th
from torch import nn

from dgl import function as fn
from dgl import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity

class DRAGConv(nn.Module):
	def __init__(
		self,
		in_feats,
		out_feats,
		num_heads,
		feat_drop=0.0,
		attn_drop=0.0,
		negative_slope=0.2,
		residual=False,
		activation=None,
		allow_zero_in_degree=False,
		bias=True,
		share_weights=False,
	):
		super(DRAGConv, self).__init__()
		self._num_heads = num_heads
		self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
		self._out_feats = out_feats
		self._allow_zero_in_degree = allow_zero_in_degree

		self.fc_src = nn.Linear(
			self._in_src_feats, out_feats * num_heads, bias=bias
		)
		if share_weights:
			self.fc_dst = self.fc_src
		else:
			self.fc_dst = nn.Linear(
				self._in_src_feats, out_feats * num_heads, bias=bias
			)
		self.fc_value = nn.Linear(
			self._in_src_feats, out_feats * num_heads, bias=bias
		)
		
		self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
		self.feat_drop = nn.Dropout(feat_drop)
		self.attn_drop = nn.Dropout(attn_drop)
		self.leaky_relu = nn.LeakyReLU(negative_slope)
		if residual:
			if self._in_dst_feats != out_feats * num_heads:
				self.res_fc = nn.Linear(
					self._in_dst_feats, num_heads * out_feats, bias=bias
				)
			else:
				self.res_fc = Identity()
		else:
			self.register_buffer("res_fc", None)
		self.activation = activation
		self.share_weights = share_weights
		self.bias = bias
		self.reset_parameters()

	def reset_parameters(self):
		"""
		Description
		-----------
		Reinitialize learnable parameters.

		Note
		----
		The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
		The attention weights are using xavier initialization method.
		"""
		gain = nn.init.calculate_gain("relu")
		nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
		nn.init.xavier_normal_(self.fc_value.weight, gain=gain)
		if self.bias:
			nn.init.constant_(self.fc_src.bias, 0)
			nn.init.constant_(self.fc_value.bias, 0)
		if not self.share_weights:
			nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
			if self.bias:
				nn.init.constant_(self.fc_dst.bias, 0)
		nn.init.xavier_normal_(self.attn, gain=gain)
		if isinstance(self.res_fc, nn.Linear):
			nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
			if self.bias:
				nn.init.constant_(self.res_fc.bias, 0)

	def set_allow_zero_in_degree(self, set_value):
		r"""
		Description
		-----------
		Set allow_zero_in_degree flag.

		Parameters
		----------
		set_value : bool
			The value to be set to the flag.
		"""
		self._allow_zero_in_degree = set_value

	def forward(self, graph, feat, get_attention=False):
		r"""
		Description
		-----------
		Compute graph attention network layer.

		Parameters
		----------
		graph : DGLGraph
			The graph.
		feat : torch.Tensor or pair of torch.Tensor
			If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
			:math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
			If a pair of torch.Tensor is given, the pair must contain two tensors of shape
			:math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
		get_attention : bool, optional
			Whether to return the attention values. Default to False.

		Returns
		-------
		torch.Tensor
			The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
			is the number of heads, and :math:`D_{out}` is size of output feature.
		torch.Tensor, optional
			The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
			edges. This is returned only when :attr:`get_attention` is ``True``.

		Raises
		------
		DGLError
			If there are 0-in-degree nodes in the input graph, it will raise DGLError
			since no message will be passed to those nodes. This will cause invalid output.
			The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
		"""
		with graph.local_scope():
			if not self._allow_zero_in_degree:
				if (graph.in_degrees() == 0).any():
					raise DGLError(
						"There are 0-in-degree nodes in the graph, "
						"output for those nodes will be invalid. "
						"This is harmful for some applications, "
						"causing silent performance regression. "
						"Adding self-loop on the input graph by "
						"calling `g = dgl.add_self_loop(g)` will resolve "
						"the issue. Setting ``allow_zero_in_degree`` "
						"to be `True` when constructing this module will "
						"suppress the check and let the code run."
					)

			h_src = h_dst = self.feat_drop(feat)
			feat_src = self.fc_src(h_src).view(
					-1, self._num_heads, self._out_feats
			)
			v_src = self.fc_value(h_src).view(
					-1, self._num_heads, self._out_feats
			)
			if self.share_weights:
					feat_dst = feat_src
			else:
					feat_dst = self.fc_dst(h_dst).view(
					-1, self._num_heads, self._out_feats
					)
			if graph.is_block:
					feat_dst = feat_dst[: graph.number_of_dst_nodes()]
					h_dst = h_dst[: graph.number_of_dst_nodes()]
					
			graph.srcdata.update(
				{"el": feat_src}
			)  # (num_src_edge, num_heads, out_dim)
			graph.srcdata.update(
				{"v": v_src}
			)  # (num_src_edge, num_heads, out_dim)
			graph.dstdata.update({"er": feat_dst})
			graph.apply_edges(fn.u_add_v("el", "er", "e"))
			e = self.leaky_relu(
				graph.edata.pop("e")
			)  # (num_src_edge, num_heads, out_dim)
			e = (
				(e * self.attn).sum(dim=-1).unsqueeze(dim=2)
			)  # (num_edge, num_heads, 1)
			# compute softmax
			graph.edata["a"] = self.attn_drop(
				edge_softmax(graph, e)
			)  # (num_edge, num_heads)
			# message passing
			graph.update_all(fn.u_mul_e("v", "a", "m"), fn.sum("m", "ft"))
			rst = graph.dstdata["ft"]
			# residual
			if self.res_fc is not None:
				resval = self.res_fc(h_dst).view(
					h_dst.shape[0], -1, self._out_feats
				)
				rst = rst + resval
			# activation
			if self.activation:
				rst = self.activation(rst)

			if get_attention:
				return rst, graph.edata["a"]
			else:
				return rst