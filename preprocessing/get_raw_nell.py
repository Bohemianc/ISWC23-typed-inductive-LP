ent2type = dict()
types=set()

with open('../data/NELL-995/train.txt', 'r') as f:
    for buf in f:
        words = buf.strip().split()
        h, t = words[0], words[2]
        type_h = h.rsplit(':', maxsplit=1)[0]
        type_t = t.rsplit(':', maxsplit=1)[0]
        types.add(type_h)
        types.add(type_t)
        if h not in ent2type:
            ent2type[h] = type_h
        if t not in ent2type:
            ent2type[t] = type_t

datasets=['nell_v1', 'nell_v2', 'nell_v3', 'nell_v4', 'nell_v1_ind', 'nell_v2_ind', 'nell_v3_ind', 'nell_v4_ind']
for dataset in datasets:
    with open(f'../data/{dataset}/train.txt', 'r') as f:
        for buf in f:
            words = buf.strip().split()
            h, t = words[0], words[2]
            type_h = h.rsplit(':', maxsplit=1)[0]
            type_t = t.rsplit(':', maxsplit=1)[0]
            types.add(type_h)
            types.add(type_t)
            if h not in ent2type:
                ent2type[h] = type_h
                print(h)
            if t not in ent2type:
                ent2type[t] = type_t
                print(t)

with open('../types/nell_ent2type_0708.txt', 'w') as f:
    for e in ent2type:
        f.write(f'{e}\t{ent2type[e]}\n')

print(len(types))