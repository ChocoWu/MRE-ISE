import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import dense_to_sparse
from cores.gene.backbone import GAT, FustionLayer, GraphLearner
from cores.lamo.decoding_network import DecoderNetwork


class MRE(nn.Module):
    def __init__(self, args, vision_config, text_config, vision_model, text_model, num_labels,
                 text_bow_size, visual_bow_size, tokenizer, processor):
        super(MRE, self).__init__()
        self.args = args
        self.hid_size = args.hid_size
        self.num_layers = args.num_layers
        self.num_labels = num_labels

        # encode image and text
        self.vision_model = vision_model
        self.text_model = text_model

        self.tokenizer = tokenizer
        self.processor = processor

        self.device = torch.device(args.device)

        # construct cross-modal graph
        self.text_linear = nn.Linear(text_config.hidden_size, args.hid_size)
        self.vision_linear = nn.Linear(vision_config.hidden_size, args.hid_size)
        self.v_abj_attr_linear = nn.Linear(text_config.hidden_size + vision_config.hidden_size, args.hid_size)
        self.text_rela_linear = nn.Linear(text_config.hidden_size, args.hid_size)
        self.vision_rela_linear = nn.Linear(text_config.hidden_size, args.hid_size)

        self.fuse = FustionLayer(args.hid_size)
        self.cross_GAT_layers = GAT(args.hid_size, args.hid_size, 0, args.dropout, alpha=0)

        # learn the best graph for MRE
        self.adjust_layers = nn.ModuleList([GAT(args.hid_size, args.hid_size, 0, args.dropout, alpha=0) for i in range(self.num_layers)])
        self.graph_learner = GraphLearner(input_size=args.hid_size, hidden_size=args.hid_size,
                                          graph_type=args.graph_type, top_k=args.top_k,
                                          epsilon=args.epsilon, num_pers=args.num_per,
                                          metric_type=args.graph_metric_type, temperature=args.temperature,
                                          feature_denoise=args.feature_denoise, device=self.device)

        self.fc_mu = nn.Linear(3 * args.hid_size, 3 * args.hid_size)  # mu var
        self.fc_logvar = nn.Linear(3 * args.hid_size, 3 * args.hid_size)

        self.topic_kl_weight = args.topic_beta
        self.topic_model = DecoderNetwork(
            text_bow_size, visual_bow_size, args.hid_size, args.inference_type, args.n_topics, args.model_type,
            args.hid_size, args.activation,
            args.dropout, args.learn_priors, label_size=args.label_size)
        self.topic_keywords_number = args.topic_keywords_number
        self.classifier1 = nn.Linear(args.hid_size, num_labels)
        self.classifier2 = nn.Linear(args.hid_size * 3 + text_config.hidden_size*3, num_labels)

    def forward(self, input_ids=None, attention_mask=None, piece2word=None,
                head_tail_pos=None,  head_object_tokens=None, tail_object_tokens=None,
                t_objects_tokens=None, t_attributes_tokens=None, t_relations_tokens=None, 
                v_objects_tokens=None,  v_attributes_tokens=None, v_relations_tokens=None,
                TSG_adj_matrix=None, TSG_edge_mask=None, VSG_adj_matrix=None, VSG_edge_mask=None,
                labels=None, writer=None, step=None,
                X_T_bow=None, X_V_bow=None):
        """
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len]
        :param piece2word: [batch_size, seq_len, seq_len]
        :param head_tail_pos: [batch_size, 4]
        :param head_object_tokens: [batch_size, head_obj_seq_len]
        :param tail_object_tokens: [batch_size, tail_obj_seq_len]
        :param t_objects_tokens: [batch_size, num_t_objects, t_obj_seq_len]
        :param t_attributes_tokens: [batch_size, num_t_objects, t_attr_seq_len]
        :param t_relations_tokens: [batch_size, num_t_relations, t_rel_seq_len]
        :param v_objects_tokens: [batch_size, num_v_objects, v_obj_seq_len]
        :param v_attributes_tokens: [batch_size, num_v_objects, v_attr_seq_len]
        :param v_relations_tokens: [batch_size,  num_v_relations, v_rel_seq_len]
        :param TSG_adj_matrix: [batch_size, num_t_objects, num_t_objects]
        :param TSG_edge_mask: [batch_size, num_t_objects, num_t_objects]
        :param VSG_adj_matrix: [batch_size, num_v_objects, num_v_objects]
        :param VSG_edge_mask: [batch_size, num_v_objects, num_v_objects]
        :param labels: [batch_size]

        """
        bsz = input_ids.size(0)
        text_hidden_state = self.text_model(input_ids, attention_mask).pooler_output

        head_obj_hidden_state = self.text_model(head_object_tokens, attention_mask=head_object_tokens.ne(0)).last_hidden_state
        tail_obj_hidden_state = self.text_model(tail_object_tokens, attention_mask=tail_object_tokens.ne(0)).last_hidden_state

        # encoding the objects and relations in TSG and VSG
        num_t_objects = t_objects_tokens.size(1)
        t_attr_seq_len = t_attributes_tokens.size(-1)
        assert t_attributes_tokens.size(1) == t_objects_tokens.size(1)
        t_objects_hidden_states = self.text_model(t_objects_tokens.view(-1, t_attr_seq_len), attention_mask=t_attributes_tokens.ne(0)).last_hidden_state
        t_objects_hidden_states = t_objects_hidden_states.sum(-2).view(bsz, num_t_objects, -1)
        num_t_relations, t_rel_seq_len = t_relations_tokens.size(1), t_relations_tokens.size(-1)
        t_relations_hidden_states = self.text_model(t_relations_tokens.view(-1, t_rel_seq_len), attention_mask=t_relations_tokens.ne(0)).last_hidden_state
        t_relations_hidden_states = t_relations_hidden_states.sum(-2).view(bsz, num_t_relations, -1)

        num_v_objects = v_objects_tokens.size(1)
        v_attr_seq_len = v_attributes_tokens.size(-1)
        assert v_attributes_tokens.size(1) == v_objects_tokens.size(1)
        v_objects_txt_hidden_states = self.text_model(v_attributes_tokens.view(-1, v_attr_seq_len)).last_hidden_state
        v_objects_txt_hidden_states = v_objects_txt_hidden_states.sum(-2).view(bsz, num_v_objects, -1)
        v_objects_img_hidden_states = self.vision_model(v_objects_tokens.view(-1, v_attr_seq_len)).pooler_output
        v_objects_hidden_states = self.v_abj_attr_linear(torch.cat([v_objects_txt_hidden_states, v_objects_img_hidden_states], dim=-1))
        num_v_relations, v_rel_seq_len = v_relations_tokens.size(1), v_relations_tokens.size(-1)
        v_relations_hidden_states = self.text_model(v_relations_tokens.view(-1, v_rel_seq_len)).last_hidden_state
        v_relations_hidden_states = v_relations_hidden_states.sum(-2).view(bsz, num_v_relations, -1)


        # encoding the objects and relations in TSG and VSG
        text_hidden_states = self.text_linear(t_objects_hidden_states)
        imgs_hidden_states = self.vision_linear(v_objects_hidden_states)
        t_relations_hidden_states = self.text_rela_linear(t_relations_hidden_states)
        v_relations_hidden_states = self.vision_rela_linear(v_relations_hidden_states)

        adj, rela = self.fuse(text_hidden_states, attention_mask, text_adj_matrix=TSG_adj_matrix, text_rela_hidden_states=t_relations_hidden_states, 
                        imgs_hidden_states=imgs_hidden_states, img_attention_mask=attention_mask, img_adj_matrix=VSG_adj_matrix, imgs_rela_hidden_states=v_relations_hidden_states,
                        threshold=self.args.threshold)
        hidden_states = torch.cat([text_hidden_states, imgs_hidden_states], dim=1)
        hidden_states = self.cross_GAT_layers(hidden_states, adj, rela)

        prior_mean, prior_variance, posterior_mean, posterior_variance, \
        posterior_log_variance, text_word_dists, visual_word_dists, estimated_labels = self.topic_model(X_T_bow,
                                                                                                        hidden_states,
                                                                                                        labels)

        # backward pass
        kl_loss, t_rl_loss, v_rl_loss = self._topic_reconstruction_loss(
            X_T_bow, X_V_bow, text_word_dists, visual_word_dists, prior_mean, prior_variance,
            posterior_mean, posterior_variance, posterior_log_variance)

        topic_loss = self.topic_kl_weight * kl_loss + t_rl_loss + v_rl_loss
        topic_loss = topic_loss.sum()

        t_node_mask = t_objects_tokens.ne(0).sum(1).ne(0)
        v_node_mask = v_objects_tokens.ne(0).sum(1).ne(0)
        node_mask = torch.cat([t_node_mask, v_node_mask], dim=-1)
        edge_mask = torch.zeros_like(adj)
        edge_mask[:, :num_t_objects, :num_t_objects] = TSG_edge_mask    
        edge_mask[:, num_t_objects:, num_t_objects:] = VSG_edge_mask
        for b in range(bsz):
            edge_mask[b, :torch.count_nonzero(t_node_mask[b]).item(), num_t_objects:torch.count_nonzero(v_node_mask[b]).item()+num_t_objects] = 1
            edge_mask[b, num_t_objects:torch.count_nonzero(v_node_mask[b]).item()+num_t_objects, :torch.count_nonzero(t_node_mask[b]).item()] = 1
        edge_mask[:, :num_t_objects, ] = 0.0
        for layer in self.adjust_layers:
            new_feature, new_adj = self.learn_graph(node_features=hidden_states,
                                                    graph_skip_conn=self.args.graph_skip_conn,
                                                    graph_include_self=self.args.graph_include_self,
                                                    init_adj=adj, node_mask=node_mask)
            adj = torch.mul(new_adj, edge_mask)
            hidden_states = layer(new_feature, adj)

        edge_number = self.cnt_edges(adj)
        writer.add_scalar(tag='edge_number', scalar_value=edge_number/bsz,
                          global_step=step)  # tensorbordx

        a = torch.mean(hidden_states, dim=1)

        entity_hidden_state = torch.Tensor(bsz, 2 * self.args.hid_size)  # batch, 2*hidden
        for i in range(bsz):
            head_idx = head_tail_pos[i][:2]
            tail_idx = head_tail_pos[i][2:]
            head_hidden = torch.max(hidden_states[i, head_idx, :], dim=0).values
            tail_hidden = torch.max(hidden_states[i, tail_idx, :], dim=0).values
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        z_hat = torch.cat([entity_hidden_state, a], dim=-1)
        mu = self.fc_mu(z_hat)
        std = F.softplus(self.fc_logvar(z_hat))
        z = self.reparametrize_n(mu, std)

        logits1 = self.classifier1(z)

        # topic integration
        topic_distribution = self.topic_model.get_theta(X_T_bow, hidden_states, labels)
        topic_label = torch.argmax(topic_distribution, dim=-1)
        visual_topic_keywords_distribution = self.topic_model.beta  # n_topics * Vocab
        textual_topic_keywords_distribution = self.topic_model.alpha
        _, T_idxs = torch.topk(torch.index_select(textual_topic_keywords_distribution, dim=0, index=topic_label),
                               self.topic_keywords_number)
        T_idxs = T_idxs.cpu().numpy()
        T_component_words = [[self.idx_2_T_token[idx] for idx in T_idxs[i]] for i in T_idxs.shape[0]]

        _, V_idxs = torch.topk(torch.index_select(visual_topic_keywords_distribution, dim=0, index=topic_label),
                               self.topic_keywords_number)
        V_idxs = V_idxs.cpu().numpy()
        V_component_words = [[self.idx_2_V_token[idx] for idx in V_idxs[i]] for i in V_idxs.shape[0]]

        T_component_words_emb = self.text_model(self.tokenizer(T_component_words)).last_hidden_state
        V_component_words_emb = self.vision_model(self.processor(V_component_words)).last_hidden_state

        T_topic_inte = self._topic_words_attention(T_component_words_emb, z)
        V_topic_inte = self._topic_words_attention(V_component_words_emb, z)

        s = torch.cat([z, T_topic_inte, V_topic_inte, text_hidden_state, head_obj_hidden_state, tail_obj_hidden_state], dim=-1)

        logits2 = self.classifier2(s)
        return (mu, std), logits1, logits2, topic_loss

    def _topic_words_attention(self, topic_words_rep, compressed_rep):
        """
        Integrate the topic keywords retrieved from the latent topic model into the compressed representation for
        enhancing the context.
        """
        _x = compressed_rep.unsqueeze(1).repeat(1, topic_words_rep.size(1), 1)
        attn_weights = torch.sigmoid(torch.sum(_x * topic_words_rep, dim=-1))
        attn_weights = attn_weights.unsqueeze(2).repeat(1, 1, topic_words_rep.size(2))
        out = torch.sum(topic_words_rep*attn_weights, dim=1)
        return out

    def _topic_reconstruction_loss(self, text_inputs, visual_inputs, text_word_dists, visual_word_dists, prior_mean,
                                   prior_variance, posterior_mean, posterior_variance, posterior_log_variance):

        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (
            var_division + diff_term - self.n_components + logvar_det_division)

        # Reconstruction term
        T_RL = -torch.sum(text_inputs * torch.log(text_word_dists + 1e-10), dim=1)

        V_RL = -torch.sum(visual_inputs * torch.log(visual_word_dists + 1e-10), dim=1)


        return KL, T_RL, V_RL

    def learn_graph(self, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None,
                    node_mask=None):
        new_feature, new_adj = self.graph_learner(node_features, node_mask=node_mask)
        bsz = node_features.size(0)
        if graph_skip_conn in (0.0, None):
            # add I
            if graph_include_self:
                if torch.cuda.is_available():
                    new_adj = new_adj + torch.stack([torch.eye(new_adj.size(1)) for _ in range(bsz)], dim=0).cuda()
                else:
                    new_adj = new_adj + torch.stack([torch.eye(new_adj.size(1)) for _ in range(bsz)], dim=0)
        else:
            # skip connection
            new_adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * new_adj

        return new_feature, new_adj

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def cnt_edges(self, adj):
        e = torch.ones_like(adj)
        o = torch.zeros_like(adj)
        a = torch.where(adj > 0.0, e, o)
        from torch_geometric.utils import remove_self_loops
        edge_number = remove_self_loops(edge_index=dense_to_sparse(a)[0])[0].size(1) / 2
        return edge_number

    def reset_parameters(self):
        self.text_linear.reset_parameters()
        self.vision_linear.reset_parameters()
        self.GAT_layer.reset_parameters()
        self.graph_learner.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
