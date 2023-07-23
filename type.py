import numpy as np
import json
import random
import os


def save_type_id(fn_ent2type, fn_type):
    id2type = []
    type2id = dict()
    with open(fn_ent2type, 'r') as f:
        for buf in f:
            words = buf.strip().split()
            for t in words[1:]:
                if t not in type2id:
                    type2id[t] = len(id2type)
                    id2type.append(t)

    with open(fn_type, 'w') as f:
        json.dump(type2id, f)

    return id2type, type2id


def get_domain_and_range(fn_dr, fn_ent2type, fn_type, relation2id):
    _, type2id = save_type_id(fn_ent2type, fn_type)

    num_rel = len(relation2id)
    rel_doms = [[] for _ in range(num_rel)]
    rel_rans = [[] for _ in range(num_rel)]

    ont_edges = set()

    with open(fn_dr, 'r') as f:
        for buf in f:
            dom, r, ran = buf.strip().split()
            if r in relation2id:
                dom_id, ran_id = type2id[dom], type2id[ran]
                r_id = relation2id[r]

                rel_doms[r_id].append(dom_id)
                rel_rans[r_id].append(ran_id)

                ont_edges.add((dom_id, r_id, ran_id))

    return rel_doms, rel_rans, ont_edges, len(type2id)


def get_ent_types(fn, ent2id, type2id):
    # print(len(ent2id))
    # print(ent2id.values())
    ent2types = [[] for _ in range(len(ent2id))]
    with open(fn, 'r') as f:
        for buf in f:
            words = buf.strip().split('\t')
            if words[0] in ent2id:
                for t in words[1:]:
                    # assert ent2id[words[0]]
                    # assert type2id[t]
                    ent2types[ent2id[words[0]]].append(type2id[t])
    # print(ent2id['/m/0gq9h'])
    # print(ent2types[600])
    return ent2types


def get_class_hier(dataset, type2id):
    dataset = dataset.split('_')[0]
    fn = f'types/{dataset}_class_hier.txt'

    if not os.path.exists(fn):
        return None

    class_hier = []

    with open(fn, 'r') as f:
        for buf in f:
            words = buf.strip().split()
            class_hier.append((type2id[words[0]], type2id[words[2]]))

    return class_hier
