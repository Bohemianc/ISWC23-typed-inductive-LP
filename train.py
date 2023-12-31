import json
import os
import argparse
import logging
import torch
import random
import numpy as np
import pickle
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.dgl.graph_classifier import GraphClassifier as dgl_model
from model.dgl.graph_classifier import TransE, TypeScorer

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter

from type import get_domain_and_range, sample_neg, get_ent_types, save_type_id, get_class_hier
from type_graph import create_type_graph


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir,
                                  f'../data/{params.dataset}/subgraphs_{params.model}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=False,
                            num_neg_samples_per_link=params.num_neg_samples_per_link)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=False,
                            num_neg_samples_per_link=params.num_neg_samples_per_link)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model, load_model=False)

    ent2types = None
    class_hier = None
    type2id = None
    ont_scorer = None
    type_scorer = None
    full_g = None
    full_rel_labels = None
    if params.ont or params.type_graph:
        dataset_prefix = params.dataset.split("_")[0]
        _, type2id = save_type_id(f'./types/{dataset_prefix}_ent2type_top.txt',
                                  f'./data/{params.dataset}/type2id.json')
        params.num_types = len(type2id)

        ent2types = get_ent_types(f'./types/{dataset_prefix}_ent2type_top.txt', train.entity2id, type2id)
        ent2types = np.array(ent2types, dtype=object)

    if params.ont:
        ont_scorer = TransE(params, graph_classifier.rel_emb).to(device=params.device)
        # class_hier = get_class_hier(params.dataset, type2id)

    print(params.type_graph)
    if params.type_graph:
        with open(f'{params.main_dir}/../data/{params.dataset}/train_neg.pkl', 'rb') as f:
            neg_triples = pickle.load(f)
        tt2ids, full_g, full_rel_labels, labels = create_type_graph(ent2types, train.triplets['train'],
                                                                    params.num_types, neg_triples)
        with open(f'./data/{params.dataset}/tt2id.pkl', 'wb') as f:
            pickle.dump(tt2ids, f)
        full_g = full_g.to(params.device)
        full_rel_labels = torch.tensor(full_rel_labels, device=params.device)
        params.num_tg_nodes = full_g.number_of_nodes()
        params.num_tg_rels = 6
        type_scorer = TypeScorer(params, ent2types, tt2ids).to(device=params.device)

    print(f"Device: {params.device}")
    print(
        f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid, ont_scorer=ont_scorer, ent2types=ent2types, type_scorer=type_scorer, full_g=full_g, full_rel_labels=full_rel_labels)

    trainer = Trainer(params, graph_classifier, train, ont_scorer=ont_scorer, ent2types=ent2types, type_scorer=type_scorer,
                      valid_evaluator=valid_evaluator, full_g=full_g, full_rel_labels=full_rel_labels)

    print('Starting training with full batch...')

    trainer.train()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--model", type=str, default="RMPI", help="model name")
    parser.add_argument("--expri_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default="toy", help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=20, help="Learning rate of the optimizer")
    parser.add_argument("--eval_every_iter", type=int, default=455,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-2, help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloading processes")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=False,
                        help='whether to only consider enclosing subgraph')
    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32, help="Relation embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--target2nei_atten', action='store_true',
                        help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--conc', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--epoch', type=int, default=0, help='to record epoch')
    parser.add_argument('--ablation', type=int, default=0,
                        help='0,1 correspond to base, NE')

    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument("--type_emb_dim", "-t_dim", type=int, default=32,
                        help="Type embedding size")
    parser.add_argument("--alpha", type=float, default=1,
                        help="The weight of ontologies.")
    parser.add_argument("--margin_o", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument('--ont', action='store_true')

    params = parser.parse_args()
    initialize_experiment(params)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.valid_file))
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
