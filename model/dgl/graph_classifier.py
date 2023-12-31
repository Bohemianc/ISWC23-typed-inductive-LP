import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.relation_list = list(self.relation2id.values())
        self.link_mode = 6
        self.is_big_dataset = False
        self.is_big_dataset = True if self.params.dataset in ['wikidata_small'] else False

        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        torch.nn.init.normal_(self.rel_emb.weight)

        self.fc_reld1 = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim, bias=True)
                                       for _ in range(6)
                                       ])
        self.fc_reld2 = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim, bias=True)
                                       for _ in range(6)
                                       ])
        self.fc_reld = nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim, bias=True)

        self.fc_layer = nn.Linear(self.params.rel_emb_dim, 1)

        if self.params.conc:
            self.conc = nn.Linear(self.params.rel_emb_dim * 2, self.params.rel_emb_dim)

        if self.params.gpu >= 0:
            self.device = torch.device('cuda:%d' % self.params.gpu)
        else:
            self.device = torch.device('cpu')

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.drop = torch.nn.Dropout(0.5)

    def rel_aggr(self, graph, u_node, v_node, num_nodes, num_edges, aggr_flag, is_drop):
        # print("node pair:", u_node, v_node)
        u_in_edge = graph.in_edges(u_node, 'all')
        u_out_edge = graph.out_edges(u_node, 'all')
        v_in_edge = graph.in_edges(v_node, 'all')
        v_out_edge = graph.out_edges(v_node, 'all')

        edge_mask = self.drop(torch.ones(num_edges))
        edge_mask = edge_mask.repeat(num_nodes, 1)

        in_edge_out = torch.sparse_coo_tensor(torch.cat((u_in_edge[1].unsqueeze(0), u_in_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(u_in_edge[2])), size=torch.Size((num_nodes, num_edges)))
        out_edge_out = torch.sparse_coo_tensor(torch.cat((u_out_edge[0].unsqueeze(0), u_out_edge[2].unsqueeze(0)), 0),
                                               torch.ones(len(u_out_edge[2])), size=torch.Size((num_nodes, num_edges)))
        in_edge_in = torch.sparse_coo_tensor(torch.cat((v_in_edge[1].unsqueeze(0), v_in_edge[2].unsqueeze(0)), 0),
                                             torch.ones(len(v_in_edge[2])), size=torch.Size((num_nodes, num_edges)))
        out_edge_in = torch.sparse_coo_tensor(torch.cat((v_out_edge[0].unsqueeze(0), v_out_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(v_out_edge[2])), size=torch.Size((num_nodes, num_edges)))

        if is_drop:
            in_edge_out = self.sparse_dense_mul(in_edge_out, edge_mask)
            out_edge_out = self.sparse_dense_mul(out_edge_out, edge_mask)
            in_edge_in = self.sparse_dense_mul(in_edge_in, edge_mask)
            out_edge_in = self.sparse_dense_mul(out_edge_in, edge_mask)

        if self.is_big_dataset:  # smaller memory
            in_edge_out = self.sparse_index_select(in_edge_out, u_node).to(device=self.device)
            out_edge_out = self.sparse_index_select(out_edge_out, u_node).to(device=self.device)
            in_edge_in = self.sparse_index_select(in_edge_in, v_node).to(device=self.device)
            out_edge_in = self.sparse_index_select(out_edge_in, v_node).to(device=self.device)
        else:  # faster calculation
            in_edge_out = in_edge_out.to(device=self.device).to_dense()[u_node].to_sparse()
            out_edge_out = out_edge_out.to(device=self.device).to_dense()[u_node].to_sparse()
            in_edge_in = in_edge_in.to(device=self.device).to_dense()[v_node].to_sparse()
            out_edge_in = out_edge_in.to(device=self.device).to_dense()[v_node].to_sparse()

        edge_mode_5 = out_edge_out.mul(in_edge_in)
        edge_mode_6 = in_edge_out.mul(out_edge_in)
        out_edge_out = out_edge_out.sub(edge_mode_5)
        in_edge_in = in_edge_in.sub(edge_mode_5)
        in_edge_out = in_edge_out.sub(edge_mode_6)
        out_edge_in = out_edge_in.sub(edge_mode_6)

        if aggr_flag == 1:
            edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]

            rel_neighbor_embd = sum([torch.sparse.mm(edge_connect_l[i],
                                                     self.fc_reld2[i](self.h1)) for i in range(self.link_mode)])

            return rel_neighbor_embd

        elif aggr_flag == 2:
            edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]

            if self.params.target2nei_atten:
                xxx = self.rel_emb(self.neighbor_edges2rels)
                rel_2directed_atten = torch.einsum('bd,nd->bn', [xxx, self.h0])
                rel_2directed_atten = self.leakyrelu(rel_2directed_atten)

                item = list()
                for i in range(6):
                    atten = self.sparse_dense_mul(edge_connect_l[i], rel_2directed_atten).to_dense()
                    mask = (atten == 0).bool()
                    atten_softmax = torch.nn.Softmax(dim=-1)(atten.masked_fill(mask, -np.inf))
                    atten_softmax = torch.where(torch.isnan(atten_softmax), torch.full_like(atten_softmax, 0),
                                                atten_softmax).to_sparse()
                    agg_i = torch.sparse.mm(atten_softmax, self.fc_reld1[i](self.h0))
                    item.append(agg_i)
                rel_neighbor_embd = sum(item)

            else:
                rel_neighbor_embd = sum([torch.sparse.mm(edge_connect_l[i],
                                                         self.fc_reld1[i](self.h0)) for i in
                                         range(self.link_mode)])

            return rel_neighbor_embd

        elif aggr_flag == 0:
            # # print(edge_mode_5)
            num_target = u_node.shape[0]
            dis_target_edge_ids = self.rel_edge_ids
            self_mask = torch.ones((num_target, num_edges))
            for i in range(num_target):
                self_mask[i][dis_target_edge_ids[i]] = 0
            self_mask = self_mask.to(device=self.device)
            edge_mode_5 = self.sparse_dense_mul(edge_mode_5, self_mask)

            edge_connect_l = in_edge_out + out_edge_out + in_edge_in + out_edge_in + edge_mode_5 + edge_mode_6

            neighbor_rel_embeds = self.rel_emb(graph.edata['type'])

            rel_2directed_atten = torch.einsum('bd,nd->bn', [self.fc_reld(self.rel_emb(self.rel_labels)),
                                                             self.fc_reld(neighbor_rel_embeds)])
            rel_2directed_atten = self.leakyrelu(rel_2directed_atten)

            atten = self.sparse_dense_mul(edge_connect_l, rel_2directed_atten).to_dense()
            mask = (atten == 0).bool()
            atten_softmax = torch.nn.Softmax(dim=-1)(atten.masked_fill(mask, -np.inf))
            atten_softmax = torch.where(torch.isnan(atten_softmax), torch.full_like(atten_softmax, 0),
                                        atten_softmax).to_sparse()
            rel_neighbor_embd = torch.sparse.mm(atten_softmax, self.fc_reld(neighbor_rel_embeds))

            return rel_neighbor_embd

    def forward(self, data):

        en_g, dis_g, rel_labels = data

        # relational aggregation begin
        self.rel_labels = rel_labels
        num_nodes = en_g.number_of_nodes()
        num_edges = en_g.number_of_edges()

        head_ids = (en_g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (en_g.ndata['id'] == 2).nonzero().squeeze(1)

        head_node, tail_node = head_ids, tail_ids
        u_in_nei = en_g.in_edges(head_node, 'all')
        u_out_nei = en_g.out_edges(head_node, 'all')
        v_in_nei = en_g.in_edges(tail_node, 'all')
        v_out_nei = en_g.out_edges(tail_node, 'all')

        edge2rel = dict()
        for i in range(len(rel_labels)):
            u_node_i = head_node[i]
            v_node_i = tail_node[i]
            u_i_in_edge = en_g.in_edges(u_node_i, 'all')[2]
            u_i_out_edge = en_g.out_edges(u_node_i, 'all')[2]
            v_i_in_edge = en_g.in_edges(v_node_i, 'all')[2]
            v_i_out_edge = en_g.out_edges(v_node_i, 'all')[2]
            i_neighbor_edges = torch.cat((u_i_in_edge, u_i_out_edge, v_i_in_edge, v_i_out_edge))
            i_neighbor_edges = torch.unique(i_neighbor_edges, sorted=False)
            # print(i_neighbor_edges)
            for eid in i_neighbor_edges.cpu().numpy().tolist():
                edge2rel[eid] = rel_labels[i]

        self.h0 = self.rel_emb(en_g.edata['type'])

        neighbor_edges = torch.cat((u_in_nei[2], u_out_nei[2], v_in_nei[2], v_out_nei[2]))
        neighbor_edges = torch.unique(neighbor_edges, sorted=False)

        neighbor_edges2rels = [edge2rel[eid] for eid in neighbor_edges.cpu().numpy().tolist()]
        neighbor_edges2rels = torch.Tensor(neighbor_edges2rels).long().to(device=self.device)

        neighbor_u_nodes = en_g.edges()[0][neighbor_edges]
        neighbor_v_nodes = en_g.edges()[1][neighbor_edges]

        self.neighbor_edges = neighbor_edges
        self.neighbor_edges2rels = neighbor_edges2rels

        '''
        agg_flag:
            2: 2-hop neighbors
            1: 1-hop directed neighbors
            0: 1-hop disclosing directed neighbors
        '''
        self.h0_extracted = self.h0[neighbor_edges]
        h_0_N = self.rel_aggr(en_g, neighbor_u_nodes, neighbor_v_nodes, num_nodes, num_edges, aggr_flag=2, is_drop=True)
        h_0_N = F.relu(h_0_N)
        self.h1 = self.rel_emb(en_g.edata['type'])

        for i, eid in enumerate(neighbor_edges):
            self.h1[eid] = self.h1[eid] + h_0_N[i]

        rel_edge_ids = [en_g.edge_id(head_ids[i], tail_ids[i], return_array=True) for i in range(head_ids.shape[0])]

        # print(self.h1.shape)
        flat_rel_edge_ids = [int(ts[0]) for ts in rel_edge_ids]
        # print(flat_rel_edge_ids)
        self.h1_extracted = self.h1[flat_rel_edge_ids]
        self.rel_edge_ids = rel_edge_ids
        self.rel_edge_ids = rel_edge_ids
        h_1_N = self.rel_aggr(en_g, head_node, tail_node, num_nodes, num_edges, aggr_flag=1, is_drop=True)

        h_1_N = F.relu(h_1_N)
        h2 = self.h1_extracted + h_1_N

        if self.params.ablation == 0:  # RMP base
            final_embed = h2
            g_rep = F.normalize(final_embed, p=2, dim=-1)
        elif self.params.ablation == 1:  # RMP NE
            # # entity aggregation begin
            dis_head_ids = (dis_g.ndata['id'] == 1).nonzero().squeeze(1)
            dis_tail_ids = (dis_g.ndata['id'] == 2).nonzero().squeeze(1)

            dis_num_nodes = dis_g.number_of_nodes()
            dis_num_edges = dis_g.number_of_edges()
            one_hop_nei_embd = self.rel_aggr(dis_g, dis_head_ids, dis_tail_ids, dis_num_nodes, dis_num_edges,
                                             aggr_flag=0, is_drop=True)
            one_hop_nei_embd = F.relu(one_hop_nei_embd)
            # #
            if self.params.conc:
                h2 = F.normalize(h2, p=2, dim=-1)
                one_hop_nei_embd = F.normalize(one_hop_nei_embd, p=2, dim=-1)
                g_rep = self.conc(torch.cat([h2, one_hop_nei_embd], dim=1))
            else:
                final_embed = h2 + one_hop_nei_embd
                g_rep = F.normalize(final_embed, p=2, dim=-1)
        # entity aggregation begin

        output = self.fc_layer(g_rep)
        return output

    @staticmethod
    def sparse_dense_mul(s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    @staticmethod
    def sparse_index_select(s, idx):
        indices_s = s._indices()
        indice_new_1 = torch.tensor([])
        indice_new_2 = torch.tensor([])
        num_i = 0.0
        for itm in idx:
            mask = (indices_s[0] == itm)
            indice_tmp_1 = torch.ones(sum(mask)) * num_i
            indice_tmp_2 = indices_s[1][mask].float()
            indice_new_1 = torch.cat((indice_new_1, indice_tmp_1), dim=0)
            indice_new_2 = torch.cat((indice_new_2, indice_tmp_2), dim=0)
            num_i = num_i + 1.0
        indices_new = torch.cat((indice_new_1.unsqueeze(0), indice_new_2.unsqueeze(0)), dim=0).long()

        return torch.sparse.FloatTensor(indices_new, torch.ones(indices_new.shape[1]),
                                        torch.Size((len(idx), s.shape[1])))


class TransE(nn.Module):
    def __init__(self, params, rel_emb):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.type_emb = nn.Embedding(self.params.num_types, self.params.type_emb_dim, sparse=False)
        # self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.rel_emb = rel_emb

    def forward(self, data, ent2types):
        g, _, rel_labels = data
        num_rels = len(rel_labels)

        ent2types = np.array(ent2types, dtype=object)

        head_ids = np.array((g.ndata['id'] == 1).nonzero().squeeze(1).cpu())
        tail_ids = np.array((g.ndata['id'] == 2).nonzero().squeeze(1).cpu())
        head_eids = g.ndata['eid'][head_ids].cpu()
        tail_eids = g.ndata['eid'][tail_ids].cpu()
        if len(head_eids) == 1:
            doms = [ent2types[head_eids]]
            rans = [ent2types[tail_eids]]
        else:
            doms = ent2types[head_eids].tolist()
            rans = ent2types[tail_eids].tolist()
        num_triples = [len(doms[i]) * len(rans[i]) for i in range(len(doms))]

        dom_ids = [doms[i][j] for i in range(len(doms)) for j in range(len(doms[i])) for _ in range(len(rans[i]))]
        ran_ids = [rans[i][j] for i in range(len(rans)) for _ in range(len(doms[i])) for j in range(len(rans[i]))]
        flat_doms = self.type_emb(torch.LongTensor(dom_ids).to(
            rel_labels.device))
        flat_rels = self.rel_emb(
            torch.LongTensor([rel_labels[i] for i in range(num_rels) for _ in range(num_triples[i])]).to(
                rel_labels.device))
        flat_rans = self.type_emb(torch.LongTensor(ran_ids).to(
            rel_labels.device))

        if torch.isnan(flat_doms).any():
            print(dom_ids)
            print(ran_ids)
            print(self.params.num_types)
            print((torch.isnan(flat_doms)).nonzero(as_tuple=True))
        assert torch.isnan(flat_doms).any() == False
        assert torch.isnan(flat_rels).any() == False
        assert torch.isnan(flat_rans).any() == False
        scores = -torch.sum(torch.pow(flat_doms + flat_rels - flat_rans, 2), dim=1)

        cur = 0
        final_scores = torch.zeros(num_rels).to(rel_labels.device)
        for i, n in enumerate(num_triples):
            if n == 0:
                final_scores[i] = 0.0
            else:
                final_scores[i] = torch.mean(scores[cur:cur + n])
                cur += n
        assert torch.isnan(final_scores).any() == False
        return final_scores
