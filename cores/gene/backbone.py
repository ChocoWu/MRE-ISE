import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.autograd import Variable
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
from torch.distributions import Bernoulli, Normal
from torch.distributions.multivariate_normal import MultivariateNormal

VERY_SMALL_NUMBER = 1e-12
INF = 1e20


class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

        self.reset_parameters()

    def forward(self, hidden_state=None, adjacent_matrix=None):
        """
        :param hidden_state: [batch_size, seq_len, hid_size]
        :param adjacent_matrix: [batch_size, seq_len, seq_len]
        """
        _x = self.linear(hidden_state)
        _x = torch.matmul(adjacent_matrix, _x)
        _x = F.relu(_x) + hidden_state
        return _x

    def reset_parameters(self):
        self.linear.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__


class GAT(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout, alpha, concat=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def forward(self, hidden_state, adjacent_matrix=None):
        """
        :param hidden_state: [batch_size, seq_len, in_features]
        :param adjacent_matrix: [batch_size, seq_len, seq_len]
        """
        _x = self.linear(hidden_state)
        e = self._para_attentional_mechanism_input(_x)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adjacent_matrix > 0.5, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, _x)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _para_attentional_mechanism_input(self, Wh):
        """
        :param Wh: [batch_size, seq_len, out_features]
        """
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class FustionLayer(nn.Module):
    def __init__(self, hid_size):
        super(FustionLayer, self).__init__()
        self.ln = nn.Sequential(nn.Linear(1 * hid_size, 1 * hid_size),
                                nn.ReLU())
        self.reset_parameters()

    def forward(self, text_obj_hidden_states=None, text_attention_mask=None, text_adj_matrix=None, 
                 imgs_obj_hidden_states=None, img_attention_mask=None, img_adj_matrix=None,
                 threshold=0.5):
        """
        build cross_modal graph.
        we build an edge between a and b only the semantic similarity sem_sim(a, b) > threshold.
        :param text_obj_hidden_states: [batch_size, num_t_objects, in_features]
        :param text_adj_matrix: [batch_size, num_v_objects, num_t_objects]
        :param text_attention_mask: [batch_size, num_t_objects]
        :param imgs_obj_hidden_states: [batch_size, num_v_objects, in-features]
        :param img_adj_matrix: [batch_size, num_v_objects, num_v_objects]
        :param img_attention_mask: [batch_size, num_v_objects]
        :param threshold: float
        """
        batch_size, num_t_objects = text_obj_hidden_states.size(0), text_obj_hidden_states.size(1)
        num_v_objects = imgs_obj_hidden_states.size(1)

        _x = self.ln(text_obj_hidden_states)
        _y = self.ln(imgs_obj_hidden_states)
        _temp = F.sigmoid(torch.bmm(_x, _y.transpose(1, 2)))

        min_value = torch.min(_temp)

        _temp = _temp.masked_fill_(1 - text_attention_mask.byte().unsqueeze(-1), min_value)
        if img_attention_mask is not None:
            _temp = _temp.masked_fill_(1 - img_attention_mask.byte().unsqueeze(1), min_value)

        max_num_nodes = num_t_objects + num_v_objects
        size = [batch_size, max_num_nodes, max_num_nodes]
        new_adj_matrix = []
        for b in range(batch_size):
            old_text_edges, old_text_edges_attr = dense_to_sparse(text_adj_matrix[b])
            new_edges = (_temp[b] > threshold).nonzero()
            head = new_edges[:, 0]
            tail = new_edges[:, 1] + num_t_objects
            if img_attention_mask is not None:
                old_img_edges, old_img_edges_attr = dense_to_sparse(img_adj_matrix[b])
                old_img_edges = [old_img_edges[i]+num_t_objects for i in range(2)]
                edge_index = torch.stack([torch.cat([old_text_edges[0], head, old_img_edges[0]]), torch.cat([old_text_edges[1], tail, old_img_edges[1]])], dim=0)
                edges_attr = torch.cat([old_text_edges_attr, _temp[b][head, tail], old_img_edges_attr], dim=0)
            else:
                edge_index = torch.stack([torch.cat([old_text_edges[0], head]), torch.cat([old_text_edges[1], tail])], dim=0)

            raw_adj = to_dense_adj(edge_index, max_num_nodes=max_num_nodes)
            new_adj_matrix.append(raw_adj)
        new_adj = torch.stack(new_adj_matrix, dim=0).squeeze().view(size)

        return new_adj

    def reset_parameters(self):
        for layer in self.ln:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, graph_type, top_k=None, epsilon=None, num_pers=1,
                 metric_type="attention", feature_denoise=True, device=None,
                 temperature=0.1):
        super(GraphLearner, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_pers = num_pers
        self.graph_type = graph_type
        self.top_k = top_k
        self.epsilon = epsilon
        self.metric_type = metric_type
        self.feature_denoise = feature_denoise
        self.temperature = temperature

        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList(
                [nn.Linear(self.input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, -num_pers))
        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, self.input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))
        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.leakyrelu = nn.LeakyReLU(0.2)
            print('[ GAT_Attention GraphLearner]')
        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(input_size, hidden_size, bias=False)
        elif metric_type == 'cosine':
            pass
        elif metric_type == 'mlp':
            self.lin = nn.Linear(self.input_size*2, 1)
        elif metric_type == 'multi_mlp':
            self.linear_sims1 = nn.ModuleList(
                [nn.Linear(self.input_size, hidden_size, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList(
                [nn.Linear(self.hidden_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))
        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(input_size, init_strategy="constant")

        print('[ Graph Learner metric type: {}, Graph Type: {} ]'.format(metric_type, self.graph_type))

    def reset_parameters(self):
        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(self.input_size, init_strategy="constant")
        if self.metric_type == 'attention':
            for module in self.linear_sims:
                module.reset_parameters()
        elif self.metric_type == 'weighted_cosine':
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        elif self.metric_type == 'gat_attention':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()
        elif self.metric_type == 'kernel':
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.init.xavier_uniform_(self.weight)
        elif self.metric_type == 'transformer':
            self.linear_sim1.reset_parameters()
            self.linear_sim2.reset_parameters()
        elif self.metric_type == 'cosine':
            pass
        elif self.metric_type == 'mlp':
            self.lin1.reset_parameters()
            self.lin2.reset_parameters()
        elif self.metric_type == 'multi_mlp':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()
        else:
            raise ValueError('Unknown metric_type: {}'.format(self.metric_type))

    def forward(self, node_features, node_mask=None):
        if self.feature_denoise:
            masked_features = self.mask_feature(node_features)
            learned_adj = self.learn_adj(masked_features, ctx_mask=node_mask)
            return masked_features, learned_adj
        else:
            learned_adj = self.learn_adj(node_features, ctx_mask=node_mask)
            return node_features, learned_adj

    def learn_adj(self, context, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)
        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """

        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0

        elif self.metric_type == 'transformer':
            Q = self.linear_sim1(context)
            attention = torch.matmul(Q, Q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            markoff_value = -INF

        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF
            # markoff_value = 0

        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis ** 2))

            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0
        elif self.metric_type == 'mlp':
            seq_len = context.size(1)
            # context_fc = torch.relu(self.lin2(torch.relu(self.lin1(context))))
            # attention = F.sigmoid(torch.matmul(context_fc, context_fc.transpose(-1, -2)))
            context_fc = context.unsqueeze(1).repeat(1, seq_len, 1, 1)
            context_bc = context.unsqueeze(2).repeat(1, 1, seq_len, 1)
            attention = F.sigmoid(self.lin(torch.cat([context_fc, context_bc], dim=-1)).squeeze())
            markoff_value = 0
        elif self.metric_type == 'multi_mlp':
            attention = 0
            for _ in range(self.num_pers):
                context_fc = torch.relu(self.linear_sims2[_](torch.relu(self.linear_sims1[_](context))))
                attention += F.sigmoid(torch.matmul(context_fc, context_fc.transpose(-1, -2)))

            attention /= self.num_pers
            markoff_value = -INF
        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.graph_type == 'epsilonNN':
            assert self.epsilon is not None
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        elif self.graph_type == 'KNN':
            assert self.top_k is not None
            attention = self.build_knn_neighbourhood(attention, self.top_k, markoff_value)
        elif self.graph_type == 'prob':
            attention = self.build_prob_neighbourhood(attention, self.epsilon, temperature=self.temperature)
        else:
            raise ValueError('Unknown graph_type: {}'.format(self.graph_type))
        if self.graph_type in ['KNN', 'epsilonNN']:
            if self.metric_type in ('kernel', 'weighted_cosine'):
                assert attention.min().item() >= 0
                attention = attention / torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            elif self.metric_type == 'cosine':
                attention = (attention > 0).float()
                attention = self.normalize_adj(attention)
            elif self.metric_type in ('transformer', 'attention', 'gat_attention'):
                attention = torch.softmax(attention, dim=-1)

        return attention

    def normalize_adj(mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def build_knn_neighbourhood(self, attention, top_k, markoff_value):
        top_k = min(top_k, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, top_k, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
        weighted_adjacency_matrix = weighted_adjacency_matrix.to(self.device)

        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        attention = torch.sigmoid(attention)
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def build_prob_neighbourhood(self, attention, epsilon=0.1, temperature=0.1):
        # attention = torch.clamp(attention, 0.01, 0.99)
        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
                                                     probs=attention).rsample()
        # eps = 0.5
        mask = (weighted_adjacency_matrix > epsilon).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def mask_feature(self, x, use_sigmoid=True, marginalize=True):
        feat_mask = (torch.sigmoid(self.feat_mask) if use_sigmoid else self.feat_mask).to(self.device)
        if marginalize:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
            z = torch.normal(mean=mean_tensor, std=std_tensor).to(self.device)
            x = x + z * (1 - feat_mask)
        else:
            x = x * feat_mask
        return x


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len, h0=None):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        if self.rnn_type == 'LSTM':
            if h0 is None:
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else:
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)



