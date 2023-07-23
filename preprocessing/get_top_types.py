def get_ent2types(dataset):
    ent2types = dict()

    with open(f'../types/{dataset}_ent2type_raw.txt', 'r') as f:
        for buf in f:
            words = buf.strip().split()
            if 'concept' in words[0]:
                ent2types[words[0]] = words[1:]
    return ent2types


#
#
# class_hier=set()
# inter_class=set()
# all_class=set()
# subclassof=dict()
#
# too_general_types=set()
# with open('types/nell_class_hier.txt', 'r') as f:
#     for buf in f:
#         c,_,p=buf.strip().split()
#         if 'thing' not in p:
#             class_hier.add((c,p))
#             inter_class.add(c)
#             all_class.add(c)
#             all_class.add(p)
#             subclassof[c]=p
#         else:
#             too_general_types.add(p)
#
# too_general_types.update(['concept:visualizablescene', 'concept:visualizableobject'])
# top_class=all_class-inter_class
#
#
# subclassof_top=dict()
# for c,p in class_hier:
#     if c not in subclassof_top:
#         while p not in top_class:
#             p=subclassof[p]
#         subclassof_top[c]=p
#     # else:
#     #     if (subclassof[c],p) in class_hier:
#     #         subclassof[c]=p
#
# # for c in subclassof_top:
# #     print(c,subclassof_top[c], subclassof_top[c] in top_class)
#
# # print(top_class)
#
# with open('types/nell_ent2type_top2.txt', 'w') as f:
#     for ent in ent2types:
#         # print(ent, end=' ')
#         tc=set()
#         for t in ent2types[ent]:
#             if not t in top_class:
#                 # print(t, end='')
#                 if not t in subclassof_top:
#                     tc.add(t)
#                     top_class.add(t)
#                 else:
#                     tc.add(subclassof_top[t])
#             else:
#                 if t not in too_general_types:
#                     tc.add(t)
#                 else:
#                     tc.add(t)
#                     top_class.add(t)
#
#         f.write(f'{ent}')
#         for t in tc:
#             f.write(f'\t{t}')
#         f.write('\n')

# read nell dom and ran
# active_types=set()
# with open('types/nell_dr.txt', 'r') as f:
#     for buf in f:
#         _,__,t = buf.strip().split()
#         if t!='concept:everypromotedthing':
#             active_types.add(t)
#
# ent2types_new={}
#
# for ent in ent2types:
#     ent2types_new[ent]=[]
#     if len(ent2types[ent])>1:
#         for t in ent2types[ent]:
#             if t in active_types:
#                 ent2types_new[ent].append(t)
#
#     elif ent2types[ent][0]!='concept:everypromotedthing':
#         ent2types_new[ent].append(ent2types[ent][0])
#         if ent2types[ent][0] not in active_types:
#             print(1)
#
# with open('types/nell_ent2type_new.txt', 'w') as f:
#     for ent in ent2types_new:
#         if len(ent2types_new[ent])>0:
#             f.write(f'{ent}')
#             for t in ent2types_new[ent]:
#                 f.write(f'\t{t}')
#             f.write('\n')

# types_fb=set()
# with open('types/fb237_ent2type_top.txt', 'r') as f:
#     for buf in f:
#         words=buf.strip().split()
#         types_fb.update(words[1:])
# print(len(types_fb))
#
# types_nell=set()
# with open('types/nell_ent2type_top.txt', 'r') as f:
#     for buf in f:
#         words=buf.strip().split()
#         types_nell.update(words[1:])
# print(len(types_nell))

def get_top_nell():
    ent2types = get_ent2types('nell')
    sup = {}
    with open(f'../types/nell_general.txt', 'r') as f:
        for buf in f:
            k, _, v = buf.strip().split()
            if v != 'concept:everypromotedthing':
                sup[k] = v

    top_types = {'concept:organization', 'concept:location', 'concept:abstractthing',
                 'concept:visualizablething', 'concept:person', 'concept:item', 'concept:university',
                 'concept:building', 'concept:mountain', 'concept:sportsteam', 'concept:park',
                 'concept:oilgasfield', 'concept:televisionstation', 'concept:lake', 'concept:street',
                 'concept:river', 'concept:city', 'concept:country', 'concept:radiostation',
                 'concept:stateorprovince'}

    added_types = set()
    for ent in ent2types:
        while ent2types[ent][0] not in top_types:
            if ent2types[ent][0] not in sup:
                added_types.add(ent2types[ent][0])
                break
            ent2types[ent][0] = sup[ent2types[ent][0]]
        # top_types.add(ent2types[ent][0])
        if 'concept' not in ent2types[ent][0]:
            print(ent2types[ent][0])
            print(ent)
    # print(ent2types)
    print(top_types)
    print(added_types)

    with open('../types/nell_ent2type_top.txt', 'w') as f:
        for ent in ent2types:
            f.write(f'{ent}\t{ent2types[ent][0]}\n')
