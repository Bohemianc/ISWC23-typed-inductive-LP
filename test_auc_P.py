# from comet_ml import Experiment
import pdb
import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning
import numpy as np

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from managers.evaluator import Evaluator

from warnings import simplefilter
import random
import pickle

from type import get_ent_types
from type_graph import create_type_graph_test
import json
from model.dgl.graph_classifier import TypeScorer


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    graph_classifier = initialize_model(params, None, load_model=True)

    logging.info(f"Device: {params.device}")

    all_auc_roc = []
    auc_roc_mean = 0

    all_auc_pr = []
    auc_pr_mean = 0
    max_label_value = np.array([2, 2])
    for r in range(1, params.runs + 1):

        params.db_path = os.path.join(params.main_dir,
                                      f'../data/{params.dataset}/test_subgraphs_{params.model}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

        generate_subgraph_datasets(params, splits=['test'],
                                   saved_relation2id=graph_classifier.relation2id,
                                   max_label_value=max_label_value)

        test = SubgraphDataset(params.db_path, 'test_pos', 'test_neg', params.file_paths, graph_classifier.relation2id,
                               add_traspose_rels=False,
                               num_neg_samples_per_link=params.num_neg_samples_per_link)

        ont_scorer = None
        type_scorer = None
        ent2types = None
        full_g = None
        full_rel_labels = None

        if params.ont or params.type_graph:
            dataset_prefix = params.dataset.rsplit('_', maxsplit=2)[0]

            type2id_path = os.path.join(params.main_dir,
                                        f'../data/{params.dataset.rsplit("_", maxsplit=1)[0]}/type2id.json')
            with open(type2id_path) as f:
                type2id = json.load(f)
            params.num_types = len(type2id)
            ent2types = get_ent_types(f'{params.main_dir}/../types/{dataset_prefix}_ent2type_top.txt',
                                      test.entity2id, type2id)
            ent2types = np.array(ent2types, dtype=object)

        if params.ont:
            ont_scorer = torch.load(os.path.join(params.exp_dir, 'best_ont_scorer.pth')).to(device=params.device)

        if params.type_graph:
            with open(f'{params.main_dir}/../data/{params.dataset.rsplit("_", maxsplit=1)[0]}/tt2id.pkl', 'rb') as f:
                old_tt2ids = pickle.load(f)
            tt2ids, full_g, full_rel_labels, old_ttids = create_type_graph_test(ent2types, test.triplets['train'],
                                                             params.num_types, old_tt2ids)
            full_g = full_g.to(params.device)
            full_rel_labels = torch.tensor(full_rel_labels, device=params.device)
            params.num_tg_nodes = len(old_tt2ids)
            params.num_tg_rels = 6
            type_scorer = TypeScorer(params, ent2types, tt2ids, is_test=True, old_ttids=old_ttids).to(device=params.device)
            type_scorer.load_state_dict(torch.load(os.path.join(params.exp_dir, 'best_type_scorer.pth')))

        test_evaluator = Evaluator(params, graph_classifier, test, ont_scorer=ont_scorer, ent2types=ent2types, type_scorer=type_scorer,
                                   full_g=full_g, full_rel_labels=full_rel_labels)

        result = test_evaluator.eval(save=True)
        print('\nTest Set Performance:' + str(result))
        all_auc_roc.append(result['auc_roc'])
        auc_roc_mean = auc_roc_mean + (result['auc_roc'] - auc_roc_mean) / r

        all_auc_pr.append(result['auc_pr'])
        auc_pr_mean = auc_pr_mean + (result['auc_pr'] - auc_pr_mean) / r

    # auc_std = np.std(all_auc_roc)
    # auc_pr_std = np.std(all_auc_pr)

    auc_roc_std = np.std(all_auc_roc)
    auc_pr_std = np.std(all_auc_pr)
    avg_auc_roc = np.mean(all_auc_roc)
    avg_auc_pr = np.mean(all_auc_pr)

    print('\nAvg test Set Performance -- mean auc_roc :' + str(avg_auc_roc) + ' std auc_roc: ' + str(auc_roc_std))
    print('\nAvg test Set Performance -- mean auc_pr :' + str(avg_auc_pr) + ' std auc_pr: ' + str(auc_pr_std))

    print(f'auc_pr: {avg_auc_pr: .4f}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--expri_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default="Toy", help="Dataset string")
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--test_file", "-t", type=str, default="test", help="Name of file containing test triplets")
    parser.add_argument("--runs", type=int, default=5, help="How many runs to perform for mean and std?")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=100000,
                        help="Set maximum number of links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloading processes")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('--target2nei_atten', action='store_true',
                        help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--conc', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--ablation', type=int, default=0, help='0,1 correspond to base, NE')

    parser.add_argument('--ont', action='store_true')
    parser.add_argument('--type_graph', '-tg', action='store_true')
    parser.add_argument("--type_emb_dim", "-t_dim", type=int, default=32,
                        help="Type embedding size")

    params = parser.parse_args()
    initialize_experiment(params)
    params.model = 'RMPI'

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'test': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.test_file))
    }
    np.random.seed(params.seed)
    random.seed(params.seed)
    torch.manual_seed(params.seed)

    if torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cudnn.deterministic = True
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
