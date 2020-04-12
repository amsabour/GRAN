import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

from classifier.module.graph_star import GraphStar

EPS = np.finfo(np.float32).eps

__all__ = ['GRANMixtureBernoulli']


class GNN(nn.Module):

    def __init__(self,
                 msg_dim,
                 node_state_dim,
                 edge_feat_dim,
                 num_prop=1,
                 num_layer=1,
                 has_attention=True,
                 att_hidden_dim=128,
                 has_residual=False,
                 has_graph_output=False,
                 output_hidden_dim=128,
                 graph_output_dim=None):
        super(GNN, self).__init__()
        self.msg_dim = msg_dim
        self.node_state_dim = node_state_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_prop = num_prop
        self.num_layer = num_layer
        self.has_attention = has_attention
        self.has_residual = has_residual
        self.att_hidden_dim = att_hidden_dim
        self.has_graph_output = has_graph_output
        self.output_hidden_dim = output_hidden_dim
        self.graph_output_dim = graph_output_dim

        self.update_func = nn.ModuleList([
            nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
            for _ in range(self.num_layer)
        ])

        self.msg_func = nn.ModuleList([
            nn.Sequential(
                *[
                    nn.Linear(self.node_state_dim + self.edge_feat_dim,
                              self.msg_dim),
                    nn.ReLU(),
                    nn.Linear(self.msg_dim, self.msg_dim)
                ]) for _ in range(self.num_layer)
        ])

        if self.has_attention:
            self.att_head = nn.ModuleList([
                nn.Sequential(
                    *[
                        nn.Linear(self.node_state_dim + self.edge_feat_dim,
                                  self.att_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.att_hidden_dim, self.msg_dim),
                        nn.Sigmoid()
                    ]) for _ in range(self.num_layer)
            ])

        if self.has_graph_output:
            self.graph_output_head_att = nn.Sequential(*[
                nn.Linear(self.node_state_dim, self.output_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.output_hidden_dim, 1),
                nn.Sigmoid()
            ])

            self.graph_output_head = nn.Sequential(
                *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

    def _prop(self, state, edge, edge_feat, layer_idx=0):
        ### compute message
        state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
        if self.edge_feat_dim > 0:
            edge_input = torch.cat([state_diff, edge_feat], dim=1)
        else:
            edge_input = state_diff

        msg = self.msg_func[layer_idx](edge_input)

        ### attention on messages
        if self.has_attention:
            att_weight = self.att_head[layer_idx](edge_input)
            msg = msg * att_weight

        ### aggregate message by sum
        state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
        scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
        state_msg = state_msg.scatter_add(0, scatter_idx, msg)

        ### state update
        state = self.update_func[layer_idx](state_msg, state)
        return state

    def forward(self, node_feat, edge, edge_feat, graph_idx=None):
        """
          N.B.: merge a batch of graphs as a single graph

          node_feat: N * D, node feature
          edge: M * 2, edge indices
          edge_feat: M * D', edge feature
          graph_idx: N * 1, graph indices
        """

        state = node_feat
        prev_state = state
        for ii in range(self.num_layer):
            if ii > 0:
                state = F.relu(state)

            for jj in range(self.num_prop):
                state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

        if self.has_residual:
            state = state + prev_state

        if self.has_graph_output:
            num_graph = graph_idx.max() + 1
            node_att_weight = self.graph_output_head_att(state)
            node_output = self.graph_output_head(state)

            # weighted average
            reduce_output = torch.zeros(num_graph,
                                        node_output.shape[1]).to(node_feat.device)
            reduce_output = reduce_output.scatter_add(0,
                                                      graph_idx.unsqueeze(1).expand(
                                                          -1, node_output.shape[1]),
                                                      node_output * node_att_weight)

            const = torch.zeros(num_graph).to(node_feat.device)
            const = const.scatter_add(
                0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

            reduce_output = reduce_output / const.view(-1, 1)

            return reduce_output
        else:
            return state


class GRANMixtureBernoulli(nn.Module):
    """ Graph Recurrent Attention Networks """

    def __init__(self, config):
        super(GRANMixtureBernoulli, self).__init__()
        self.config = config
        self.device = config.device
        self.max_num_nodes = config.model.max_num_nodes
        self.hidden_dim = config.model.hidden_dim
        self.is_sym = config.model.is_sym
        self.block_size = config.model.block_size
        self.sample_stride = config.model.sample_stride
        self.num_GNN_prop = config.model.num_GNN_prop
        self.num_GNN_layers = config.model.num_GNN_layers
        self.edge_weight = config.model.edge_weight if hasattr(
            config.model, 'edge_weight') else 1.0
        self.dimension_reduce = config.model.dimension_reduce
        self.has_attention = config.model.has_attention
        self.num_canonical_order = config.model.num_canonical_order
        self.output_dim = 1
        self.num_mix_component = config.model.num_mix_component
        self.has_rand_feat = False  # use random feature instead of 1-of-K encoding
        self.att_edge_dim = 64

        self.output_theta = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component))

        self.output_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component),
            nn.LogSoftmax(dim=1))

        # Node Representation (In the paper)
        if self.dimension_reduce:
            self.embedding_dim = config.model.embedding_dim
            self.decoder_input = nn.Sequential(
                nn.Linear(self.max_num_nodes, self.embedding_dim))
        else:
            self.embedding_dim = self.max_num_nodes

        self.decoder = GNN(
            msg_dim=self.hidden_dim,
            node_state_dim=self.hidden_dim,
            edge_feat_dim=2 * self.att_edge_dim,
            num_prop=self.num_GNN_prop,
            num_layer=self.num_GNN_layers,
            has_attention=self.has_attention)

        ### Loss functions
        pos_weight = torch.ones([1]) * self.edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction='none')

        # Graph class representation
        self.class_representation = nn.Embedding(2, self.embedding_dim)

    def _inference(self,
                   A_pad=None,
                   edges=None,
                   node_idx_gnn=None,
                   node_idx_feat=None,
                   att_idx=None):
        """ generate adj in row-wise auto-regressive fashion """

        B, C, N_max, _ = A_pad.shape
        H = self.hidden_dim
        K = self.block_size
        A_pad = A_pad.view(B * C * N_max, -1)

        if self.dimension_reduce:
            node_feat = self.decoder_input(A_pad)  # BCN_max * H
        else:
            node_feat = A_pad  # BCN_max * N_max

        ### GNN inference
        # pad zero as node feature for newly generated nodes (1st row)
        node_feat = F.pad(
            node_feat, (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) * N_max

        # N * 1 (one hot encoding of newly generated nodes)
        # create symmetry-breaking edge feature for the newly generated nodes
        att_idx = att_idx.view(-1, 1)

        if self.has_rand_feat:
            # create random feature
            att_edge_feat = torch.zeros(edges.shape[0],
                                        2 * self.att_edge_dim).to(node_feat.device)
            idx_new_node = (att_idx[[edges[:, 0]]] >
                            0).long() + (att_idx[[edges[:, 1]]] > 0).long()
            idx_new_node = idx_new_node.byte().squeeze()
            att_edge_feat[idx_new_node, :] = torch.randn(
                idx_new_node.long().sum(),
                att_edge_feat.shape[1]).to(node_feat.device)
        else:
            # create one-hot feature
            att_edge_feat = torch.zeros(edges.shape[0],
                                        2 * self.att_edge_dim).to(node_feat.device)
            # scatter with empty index seems to cause problem on CPU but not on GPU

            # E * (2 * att_edge_dim)
            edges = edges.long()
            att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
            att_edge_feat = att_edge_feat.scatter(
                1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

        # GNN inference
        # N.B.: node_feat is shared by multiple subgraphs within the same batch
        # This basically turns h_i^0 to h_i^R in the paper
        node_state = self.decoder(
            node_feat[node_idx_feat], edges, edge_feat=att_edge_feat)

        ### Pairwise predict edges
        diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]

        log_theta = self.output_theta(diff)  # B * (tt+K)K
        log_alpha = self.output_alpha(diff)  # B * (tt+K)K
        log_theta = log_theta.view(-1, self.num_mix_component)  # B * CN(N-1)/2 * K
        log_alpha = log_alpha.view(-1, self.num_mix_component)  # B * CN(N-1)/2 * K

        return log_theta, log_alpha

    def _sampling(self, B):
        """ generate adj in row-wise auto-regressive fashion """

        K = self.block_size  # 1
        S = self.sample_stride  # 1
        H = self.hidden_dim  # 256
        N = self.max_num_nodes  # 3530
        mod_val = (N - K) % S
        if mod_val > 0:
            N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
        else:
            N_pad = N

        """ First dimension is for graph number (which graph we are doing this for)
                    others are the adjacency matrix probably """
        A = torch.zeros(B, N_pad, N_pad).to(self.device)
        dim_input = self.embedding_dim if self.dimension_reduce else self.max_num_nodes  # 256

        ### cache node state for speed up
        node_state = torch.zeros(B, N_pad, dim_input).to(self.device)

        for ii in tqdm(range(0, N_pad, S)):
            jj = ii + K
            if jj > N_pad:
                break

            A = self.generate_one_block(A, ii, node_state=node_state, is_sym=False, sample=True)

        ### make it symmetric
        if self.is_sym:
            A = torch.tril(A, diagonal=-1)
            A = A + A.transpose(1, 2)

        return A

    def generate_one_block(self, A, row, node_state=None, is_sym=True, sample=False,
                           inject_graph_label=False, class_label=None):
        """
        Generate one block of nodes
        :param A: Adjacency matrix of the graph so far
        :param row: Which row are we generating
        :param node_state: State of the graph after the linear transformation
        :param is_sym: Should i symmetricize?
        :param sample: Return bernoulli probabilities or sample using them
        :param inject_graph_label: Inject the graph class representation into the GNN
        :param class_label: Graph class must be given if inject_graph_label is True
        """
        B = A.shape[0]
        K = self.block_size  # 1
        S = self.sample_stride  # 1
        H = self.hidden_dim  # 256
        N = self.max_num_nodes  # 3530
        mod_val = (N - K) % S
        if mod_val > 0:
            N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
        else:
            N_pad = N

        dim_input = self.embedding_dim if self.dimension_reduce else self.max_num_nodes  # 256

        ii = row
        jj = row + K

        # Cannot use inline operation (doesn't work with backwards)

        if inject_graph_label:
            new_A = torch.zeros((B, N + 1, N + 1))
            new_A[:, 0, :] = 1  # Add one node at the beginning connected to everything (specifies the graph label)
            new_A[:, 1:ii + 1, :] = A[:, :ii, :]
            A = new_A
            ii += 1
            jj += 1
        else:
            new_A = torch.zeros_like(A)
            new_A[:, 0, :] = 1  # Add one
            new_A[:, :ii, :] = A[:, :ii, :]
            A = new_A

        A = torch.tril(A, diagonal=-1)  # Get lower triangle

        if node_state is None:
            node_state = torch.zeros(B, N_pad, dim_input).to(self.device)
            if self.dimension_reduce:
                node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
            else:
                node_state[:, :ii, :] = A[:, ii - S:ii, :N]
        else:
            if ii >= K:
                if self.dimension_reduce:
                    node_state[:, ii - K:ii, :] = self.decoder_input(A[:, ii - K:ii, :N])
                else:
                    node_state[:, ii - K:ii, :] = A[:, ii - S:ii, :N]
            else:
                if self.dimension_reduce:
                    node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
                else:
                    node_state[:, :ii, :] = A[:, ii - S:ii, :N]

        # TODO: What is this padding for???
        node_state_in = F.pad(
            node_state[:, :ii, :], (0, 0, 0, K), 'constant', value=.0)

        ### GNN propagation
        adj = F.pad(
            A[:, :ii, :ii], (0, K, 0, K), 'constant', value=1.0)  # B * jj * jj
        adj = torch.tril(adj, diagonal=-1)
        adj = adj + adj.transpose(1, 2)
        edges = [
            adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
            for bb in range(B)
        ]
        edges = torch.cat(edges, dim=1).t()

        att_idx = torch.cat([torch.zeros(ii).long(),
                             torch.arange(1, K + 1)]).to(self.device)
        att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

        if self.has_rand_feat:
            # create random feature
            att_edge_feat = torch.zeros(edges.shape[0],
                                        2 * self.att_edge_dim).to(self.device)
            idx_new_node = (att_idx[[edges[:, 0]]] >
                            0).long() + (att_idx[[edges[:, 1]]] > 0).long()
            idx_new_node = idx_new_node.byte().squeeze()
            att_edge_feat[idx_new_node, :] = torch.randn(
                idx_new_node.long().sum(), att_edge_feat.shape[1]).to(self.device)
        else:
            # create one-hot feature
            att_edge_feat = torch.zeros(edges.shape[0],
                                        2 * self.att_edge_dim).to(self.device)
            att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
            att_edge_feat = att_edge_feat.scatter(
                1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

        # An absolute disgusting way of injecting graph_labels
        node_state_in = node_state_in.view(-1, H)
        if inject_graph_label:
            assert class_label is not None
            class_representation = self.class_representation(class_label)
            new_node_state_in = torch.zeros_like(node_state_in)
            new_node_state_in[0] = class_representation[:]
            new_node_state_in[1:] = node_state_in[1:]
            node_state_in = new_node_state_in

        node_state_out = self.decoder(
            node_state_in, edges, edge_feat=att_edge_feat)

        if inject_graph_label:
            node_state_out = node_state_out[1:]
            ii -= 1
            jj -= 1

        node_state_out = node_state_out.view(B, jj, -1)

        idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
        idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
        idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)

        diff = node_state_out[:, idx_row, :] - node_state_out[:, idx_col, :]  # B * (ii+K)K * H
        diff = diff.view(-1, node_state.shape[2])
        log_theta = self.output_theta(diff)
        log_alpha = self.output_alpha(diff)

        log_theta = log_theta.view(B, -1, K, self.num_mix_component)  # B * K * (ii+K) * L
        log_theta = log_theta.transpose(1, 2)  # B * (ii+K) * K * L

        log_alpha = log_alpha.view(B, -1, self.num_mix_component)  # B * K * (ii+K)
        prob_alpha = log_alpha.mean(dim=1).exp()
        alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

        prob = []
        for bb in range(B):
            prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]

        prob = torch.stack(prob, dim=0)

        if sample:
            A[:, ii:jj, :jj] = torch.bernoulli(prob[:, :jj - ii, :])
        else:
            new_A = torch.zeros_like(A)
            new_A[:, :, jj:] = A[:, :, jj:]
            new_A[:, :ii, :] = A[:, :ii, :]
            new_A[:, jj:, :] = A[:, jj:, :]
            new_A[:, ii:jj, jj:] = A[:, ii:jj, jj:]
            new_A[:, ii:jj, :jj] = prob[:, :jj - ii, :]
            A = new_A

            # A = prob[:, :jj - ii, :]

        # make it symmetric
        if is_sym:
            A = torch.tril(A, diagonal=-1)
            A = A + A.transpose(1, 2)

        return A

    def forward(self, input_dict):
        """
          TODO: What the hell is the difference between B and N then???????????????
          B: batch size
          N: number of rows/columns in mini-batch
          N_max: max number of rows/columns
          M: number of augmented edges in mini-batch
          H: input dimension of GNN
          K: block size
          E: number of edges in mini-batch
          S: stride
          C: number of canonical orderings
          D: number of mixture Bernoulli

          Args:
            A_pad: B * C * N_max * N_max, padded adjacency matrix
            node_idx_gnn: M * 2, node indices of augmented edges
            node_idx_feat: N * 1, node indices of subgraphs for indexing from feature
                          (0 indicates indexing from 0-th row of feature which is
                            always zero and corresponds to newly generated nodes)
            att_idx: N * 1, one-hot encoding of newly generated nodes
                          (0 indicates existing nodes, 1-D indicates new nodes in
                            the to-be-generated block)
            subgraph_idx: E * 1, indices corresponding to augmented edges
                          (representing which subgraph in mini-batch the augmented
                          edge belongs to)
            edges: E * 2, edge as [incoming node index, outgoing node index]
            label: E * 1, binary label of augmented edges
            num_nodes_pmf: N_max, empirical probability mass function of number of nodes

          Returns:
            loss                        if training
            list of adjacency matrices  else
        """
        is_sampling = input_dict[
            'is_sampling'] if 'is_sampling' in input_dict else False
        batch_size = input_dict[
            'batch_size'] if 'batch_size' in input_dict else None
        A_pad = input_dict['adj'] if 'adj' in input_dict else None
        node_idx_gnn = input_dict[
            'node_idx_gnn'] if 'node_idx_gnn' in input_dict else None
        node_idx_feat = input_dict[
            'node_idx_feat'] if 'node_idx_feat' in input_dict else None
        att_idx = input_dict['att_idx'] if 'att_idx' in input_dict else None
        subgraph_idx = input_dict[
            'subgraph_idx'] if 'subgraph_idx' in input_dict else None
        edges = input_dict['edges'] if 'edges' in input_dict else None
        label = input_dict['label'] if 'label' in input_dict else None
        num_nodes_pmf = input_dict[
            'num_nodes_pmf'] if 'num_nodes_pmf' in input_dict else None
        graph_label = input_dict['graph_label'] if 'graph_label' in input_dict else None
        graph_classifier = input_dict['graph_classifier'] if 'graph_classifier' in input_dict else None
        N_max = self.max_num_nodes

        torch.autograd.set_detect_anomaly(True)
        if not is_sampling:
            B, _, N, _ = A_pad.shape

            ### compute adj loss
            log_theta, log_alpha = self._inference(
                A_pad=A_pad,
                edges=edges,
                node_idx_gnn=node_idx_gnn,
                node_idx_feat=node_idx_feat,
                att_idx=att_idx)

            num_edges = log_theta.shape[0]

            adj_loss = mixture_bernoulli_loss(label, log_theta, log_alpha,
                                              self.adj_loss_func, subgraph_idx)
            adj_loss = adj_loss * float(self.num_canonical_order)

            graph_label = graph_label.long()

            ############ We can create an extra block like so #####################
            new_elements = (att_idx == 1).nonzero().squeeze()
            iis = node_idx_feat[new_elements - 1].cpu()
            generated_A = self.generate_one_block(A_pad[:, 0], iis, inject_graph_label=True, class_label=graph_label)[0,
                          :iis + 1, :iis + 1]

            lower_part = torch.tril(generated_A, diagonal=-1)
            x = torch.zeros((iis + 1, 3)).to('cuda')
            edge_mask = (lower_part != 0).to('cuda')
            edge_index = edge_mask.nonzero().transpose(0, 1).to('cuda')
            edge_attr = torch.masked_select(lower_part, edge_mask).to('cuda')
            batch = torch.zeros(iis + 1).long().to('cuda')

            logits_node, logits_star, logits_lp = \
                graph_classifier(x, edge_index, batch, star=None, edge_type=None, edge_attr=edge_attr)

            loss = graph_classifier.gc_loss(logits_star, graph_label)
            #######################################################################

            adj_loss += loss

            return adj_loss
        else:
            # Samples batch_size graphs of maximum size
            A = self._sampling(batch_size)

            # Pick the number of nodes of each graph based on the pmf provided
            num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.device)
            num_nodes = torch.multinomial(
                num_nodes_pmf, batch_size, replacement=True) + 1  # shape B * 1

            # Select only the first num_nodes vertices of the maximum size graph
            A_list = [
                A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
            ]
            return A_list


def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx):
    """
      Compute likelihood for mixture of Bernoulli model

      Args:
        label: E * 1, see comments above
        log_theta: E * D, see comments above
        log_alpha: E * D, see comments above
        adj_loss_func: BCE loss
        subgraph_idx: E * 1, see comments above

      Returns:
        loss: negative log likelihood
    """

    num_subgraph = subgraph_idx.max() + 1
    K = log_theta.shape[1]
    adj_loss = torch.stack(
        [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)

    const = torch.zeros(num_subgraph).to(label.device)
    const = const.scatter_add(0, subgraph_idx,
                              torch.ones_like(subgraph_idx).float())

    reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
    reduce_adj_loss = reduce_adj_loss.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)

    reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
    reduce_log_alpha = reduce_log_alpha.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
    reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)

    log_prob = -reduce_adj_loss + reduce_log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1)
    loss = -log_prob.sum() / float(log_theta.shape[0])

    return loss
