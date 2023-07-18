import argparse


def get_ent2types(dataset):
    dataset = dataset.split('_')[0]
    ent2types = dict()

# TODO: modify ent2type file
    with open(f'../types/{dataset}_ent2type_top.txt', 'r') as f:
        for buf in f:
            words = buf.strip().split()
            ent2types[words[0]] = words[1:]
    return ent2types


def cal_type_consistency(dataset, num_neg=50, k=5):
    ent2types = get_ent2types(dataset)
    with open(f'../data/{dataset}/{params.ont}_ranking_head_predictions.txt') as f:
        num_examples = 0
        hit = 0
        while True:
            line = f.readline()
            scores = []
            num_examples += 1
            if line == '':
                break

            true_s, true_r, true_o, true_v = line.strip().split()
            scores.append(((true_s, true_r, true_o), float(true_v)))
            for i in range(num_neg - 1):
                buf = f.readline()
                s, r, o, v = buf.strip().split()
                scores.append(((s, r, o), float(v)))

            scores = sorted(scores, key=lambda score: -score[1])

            if true_o not in ent2types:
                hit += k
                continue

            # print(f'{true_s},{true_r},{true_o}\t{ent2types[true_o]}\t{true_v}')
            for item in scores[:k]:
                pred_o = item[0][2]
                if pred_o not in ent2types:
                    hit += 1
                    continue
                # print(f'{pred_o}\t{ent2types[pred_o]}\t{item[1]}')

                if len(set(ent2types[true_o]) & set(ent2types[pred_o])) > 0:
                    hit += 1

    head_ratio = hit / (k * num_examples)
    # print(head_ratio)

    with open(f'../data/{dataset}/{params.ont}_ranking_tail_predictions.txt') as f:
        num_examples = 0
        hit = 0
        while True:
            line = f.readline()
            scores = []
            num_examples += 1
            if line == '':
                break

            true_s, true_r, true_o, true_v = line.strip().split()
            scores.append(((true_s, true_r, true_o), float(true_v)))
            for i in range(num_neg - 1):
                buf = f.readline()
                s, r, o, v = buf.strip().split()
                scores.append(((s, r, o), float(v)))

            scores = sorted(scores, key=lambda score: -score[1])

            if true_s not in ent2types:
                hit += k
                continue

            # print(f'{true_s},{true_r},{true_o}\t{ent2types[true_s]}\t{true_v}')
            for item in scores[:k]:
                pred_s = item[0][0]
                if pred_s not in ent2types:
                    hit += 1
                    continue
                # print(f'{pred_s}\t{ent2types[pred_s]}\t{item[1]}')

                if len(set(ent2types[true_s]) & set(ent2types[pred_s])) > 0:
                    hit += 1

        tail_ratio = hit / (k * num_examples)

    print(f'{k}\t{(head_ratio + tail_ratio) / 2:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransE model')
    # parser.add_argument("--k", "-k", type=int)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument('--ont', action='store_true')
    params = parser.parse_args()
    for k in [1,2,3,5,10]:
        cal_type_consistency(params.dataset, k=k)
