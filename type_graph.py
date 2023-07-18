from dgl import DGLGraph
import torch
import random


def sample_neg(typedtriples, num_nodes):
    neg_typedtriples = set()

    for i in range(len(typedtriples)):
        h, t, r = typedtriples[i]
        if random.random() < 0.5:
            neg_u = random.randint(0, num_nodes - 1)
            while (neg_u, r, t) in typedtriples:
                neg_u = random.randint(0, num_nodes - 1)
            neg_typedtriples.add((neg_u, r, t))
        else:
            neg_v = random.randint(0, num_nodes - 1)
            while (h, r, neg_v) in typedtriples:
                neg_v = random.randint(0, num_nodes - 1)
            neg_typedtriples.add((h, r, neg_v))

    return neg_typedtriples


def get_ntt_from_neg_triples(neg_triples, pos_tt, ent2types):
    ntt=set()
    for triple in neg_triples:
        h, t, r = triple
        c_h = ent2types[h][0]
        c_t = ent2types[t][0]
        tt = (c_h, r, c_t)
        if tt not in pos_tt:
            ntt.add(tt)
    return ntt


def create_type_graph(ent2types, triplets, num_classes, neg_triples=None):
    typetriple2id = {}
    # num_ent = len(ent2types)
    # e2tt_h = [[] for _ in range(num_ent)]
    # e2tt_t = [[] for _ in range(num_ent)]
    class2tt_dom = [[] for _ in range(num_classes)]
    class2tt_ran = [[] for _ in range(num_classes)]

    labels = []
    cnt = 0
    for triple in triplets:
        h, t, r = triple
        c_h = ent2types[h][0]
        # print(t)
        # print(len(ent2types))
        # print(ent2types[18])
        c_t = ent2types[t][0]
        tt = (c_h, r, c_t)
        if tt not in typetriple2id:
            typetriple2id[tt] = cnt
            class2tt_dom[c_h].append(cnt)
            class2tt_ran[c_t].append(cnt)
            labels.append(1)
            cnt += 1

        # e2tt_t[t].append(cnt)
        # e2tt_h[h].append(cnt)

    # neg_typedtriples = sample_neg(set(typetriple2id.keys()))
    # add neg typed triples
    if neg_triples is not None:
        pos_typedtriples = set(typetriple2id.keys())
        neg_typedtriples = get_ntt_from_neg_triples(neg_triples, pos_typedtriples, ent2types)
        # add negative typedtriples
        for ntt in neg_typedtriples:
            c_h, r, c_t = ntt
            typetriple2id[ntt] = cnt
            class2tt_dom[c_h].append(cnt)
            class2tt_ran[c_t].append(cnt)
            labels.append(0)
            cnt += 1

    id2typetriple = {typetriple2id[tt]: tt for tt in typetriple2id}
    edges = set()
    # T-H,0; H-T,1; T-T,2; H-H,3; LOOP,4; Another,5
    for i in range(num_classes):
        heads = class2tt_dom[i]
        tails = class2tt_ran[i]
        edges.update([(h, 0, t) for h in heads for t in tails])
        edges.update([(t, 1, h) for h in heads for t in tails])
        edges.update([(h, 2, t) for h in tails for t in tails if h != t])
        edges.update([(h, 3, t) for h in heads for t in heads if h != t])

        for h in heads:
            for t in tails:
                if id2typetriple[h][0] == id2typetriple[t][2]:
                    edges.add((h, 4, t))
        for h in heads:
            for t in heads:
                if h != t and id2typetriple[h][0] == id2typetriple[t][0]:
                    edges.add((h, 5, t))
                    edges.add((t, 5, h))

    edge_type = []
    edge_u, edge_v = [], []
    for h, r, t in edges:
        edge_type.append(r)
        edge_u.append(h)
        edge_v.append(t)

    edge_type = torch.tensor(edge_type)
    edge_u = torch.tensor(edge_u)
    edge_v = torch.tensor(edge_v)
    labels=torch.tensor(labels)

    type_g = DGLGraph((edge_u, edge_v))
    print(type_g)

    return typetriple2id, type_g, edge_type, labels
    # type_g.edata.update({'rel_type':edge_type})


def create_type_graph_test(ent2types, triplets, num_classes, emb_tt2ids):
    typetriple2id = {}
    class2tt_dom = [[] for _ in range(num_classes)]
    class2tt_ran = [[] for _ in range(num_classes)]

    # labels = []
    cnt = 0
    old_ttids= []
    for triple in triplets:
        h, t, r = triple
        c_h = ent2types[h][0]
        # print(t)
        # print(len(ent2types))
        # print(ent2types[18])
        c_t = ent2types[t][0]
        tt = (c_h, r, c_t)
        if tt not in typetriple2id:
            typetriple2id[tt] = cnt
            class2tt_dom[c_h].append(cnt)
            class2tt_ran[c_t].append(cnt)
            assert tt in emb_tt2ids
            old_ttids.append(emb_tt2ids[tt] if tt in emb_tt2ids else -1)
            # labels.append(1)
            cnt += 1

        # e2tt_t[t].append(cnt)
        # e2tt_h[h].append(cnt)

    # neg_typedtriples = sample_neg(set(typetriple2id.keys()))
    # add neg typed triples
    # if neg_triples is not None:
    #     pos_typedtriples = set(typetriple2id.keys())
    #     neg_typedtriples = get_ntt_from_neg_triples(neg_triples, pos_typedtriples, ent2types)
    #     # add negative typedtriples
    #     for ntt in neg_typedtriples:
    #         c_h, r, c_t = ntt
    #         typetriple2id[ntt] = cnt
    #         class2tt_dom[c_h].append(cnt)
    #         class2tt_ran[c_t].append(cnt)
    #         labels.append(0)
    #         cnt += 1

    id2typetriple = {typetriple2id[tt]: tt for tt in typetriple2id}
    edges = set()
    # T-H,0; H-T,1; T-T,2; H-H,3; LOOP,4; Another,5
    for i in range(num_classes):
        heads = class2tt_dom[i]
        tails = class2tt_ran[i]
        edges.update([(h, 0, t) for h in heads for t in tails])
        edges.update([(t, 1, h) for h in heads for t in tails])
        edges.update([(h, 2, t) for h in tails for t in tails if h != t])
        edges.update([(h, 3, t) for h in heads for t in heads if h != t])

        for h in heads:
            for t in tails:
                if id2typetriple[h][0] == id2typetriple[t][2]:
                    edges.add((h, 4, t))
        for h in heads:
            for t in heads:
                if h != t and id2typetriple[h][0] == id2typetriple[t][0]:
                    edges.add((h, 5, t))
                    edges.add((t, 5, h))

    edge_type = []
    edge_u, edge_v = [], []
    for h, r, t in edges:
        edge_type.append(r)
        edge_u.append(h)
        edge_v.append(t)

    edge_type = torch.tensor(edge_type)
    edge_u = torch.tensor(edge_u)
    edge_v = torch.tensor(edge_v)
    # labels=torch.tensor(labels)

    type_g = DGLGraph((edge_u, edge_v))
    print(type_g)

    return typetriple2id, type_g, edge_type, old_ttids

