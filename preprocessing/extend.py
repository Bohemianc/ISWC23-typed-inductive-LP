types = set()

# def get_ent2types(dataset):
#     ent2types=dict()
#     with open(f'../types/{dataset}_ent2type_raw.txt', 'r') as f:
#         for buf in f:
#             e, t=buf.strip().split()
#             types.add(t)
#             ent2types[e]=set([t])
#     return ent2types

# print(types)
class_hier = []


def read_class_hie(dataset):
    subclassof = dict()
    with open(f'../types/{dataset}_class_hier.txt', 'r') as f:
        for buf in f:
            words = buf.strip().split()
            if words[0] not in subclassof:
                subclassof[words[0]] = []
            subclassof[words[0]].append(words[2])
    return subclassof


def extend_classes(dataset):
    subclassof = read_class_hie()
    num = 0
    with open(f'../types/{dataset}_ent2type_raw.txt', 'r') as f, open(f'../types/{dataset}_ent2type_extend.txt',
                                                                      'w') as f2:
        for buf in f:
            words = buf.strip().split()
            classes = set()
            classes.update(words[1:])
            cur_classes = set()
            cur_classes.update(words[1:])

            while len(cur_classes) > 0:
                for cla in cur_classes:
                    new_classes = set()
                    if cla in subclassof:
                        classes.update(subclassof[cla])
                        new_classes.update(subclassof[cla])

                    cur_classes = new_classes

            num += len(classes) - len(words) + 1

            f2.write(f'{words[0]}')
            # print(classes)
            for cla in classes:
                f2.write(f'\t{cla}')
            f2.write('\n')

        print(num)


extend_classes('nell')
extend_classes('fb237')
