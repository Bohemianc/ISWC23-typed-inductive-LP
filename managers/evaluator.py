import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator():
    def __init__(self, params, graph_classifier, data, ont_scorer=None, ent2types=None, type_scorer=None, full_g=None, full_rel_labels=None):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

        self.ont_scorer = ont_scorer
        self.ent2types = ent2types

        self.type_scorer = type_scorer
        self.full_g = full_g
        self.full_rel_labels = full_rel_labels

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch,
                                                                                                self.params.device)

                # score_pos, _ = self.graph_classifier(data_pos)
                # score_neg, _ = self.graph_classifier(data_neg)
                score_pos = self.graph_classifier(data_pos).squeeze(1)
                score_neg = self.graph_classifier(data_neg).squeeze(1)
                if self.params.ont:
                    os_pos = self.ont_scorer(data_pos, self.ent2types)
                    os_neg = self.ont_scorer(data_neg, self.ent2types)
                    score_pos += os_pos
                    score_neg += os_neg

                if self.params.type_graph:
                    type_score_pos = self.type_scorer(data_pos, self.full_g, self.full_rel_labels)
                    type_score_neg = self.type_scorer(data_neg, self.full_g, self.full_rel_labels)
                    score_pos += type_score_pos
                    score_neg += type_score_neg

                pos_scores += score_pos.detach().cpu().tolist()
                neg_scores += score_neg.detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()

        # acc = metrics.accuracy_score(labels, preds)
        # print(pos_labels+neg_labels)
        # print(pos_scores+neg_scores)
        auc_roc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        return {'auc_roc': auc_roc, 'auc_pr': auc_pr}
