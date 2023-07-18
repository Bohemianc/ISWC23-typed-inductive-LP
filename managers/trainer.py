import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, ent2types=None, class_hier=None, type_scorer=None, full_g=None, full_rel_labels=None,
                 ont_scorer=None, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.ent2types = ent2types
        self.class_hier = class_hier
        self.ont_scorer = ont_scorer
        self.type_scorer = type_scorer
        self.full_g = full_g
        self.full_rel_labels = full_rel_labels

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        if params.ont:
            model_params+= list(self.ont_scorer.parameters())
        if params.type_graph:
            model_params+=list(self.type_scorer.parameters())
        print('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum,
                                       weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        self.criterion_ont = nn.MarginRankingLoss(self.params.margin_o, reduction='sum')
        self.criterion_bce = nn.BCELoss()
        self.cls = nn.CrossEntropyLoss()

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        total_cls_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        for b_idx, batch in enumerate(dataloader):

            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()

            # score_pos, rel_cls_embed = self.graph_classifier(data_pos)
            # score_neg, _ = self.graph_classifier(data_neg)

            score_pos = self.graph_classifier(data_pos).squeeze(1)
            score_neg = self.graph_classifier(data_neg).squeeze(1)

            if self.params.type_graph:
                type_score_pos = self.type_scorer(data_pos, self.full_g, self.full_rel_labels)
                type_score_neg = self.type_scorer(data_neg, self.full_g, self.full_rel_labels)
                # print(type_score_pos.shape)
                # print(score_pos.shape)
                score_pos += type_score_pos
                score_neg += type_score_neg
            # loss += self.type_scorer.loss(data_pos)
            # print(rel_cls_embed.shape)
            # print(cls_labels.shape)
            # loss_cls = self.cls(rel_cls_embed, data_pos[2]) * 100
            if self.params.ont:
                ont_pos = self.ont_scorer(data_pos, self.ent2types)
                ont_neg = self.ont_scorer(data_neg, self.ent2types)
                score_pos += ont_pos
                score_neg += ont_neg
                # loss += self.params.alpha * self.criterion_ont(ont_pos.view(-1, 1), ont_neg.view(-1, 1),
                #                                                torch.Tensor([1]).repeat(score_pos.size(0)).view(-1,
                #                                                                                                 1).to(device=self.params.device))

            loss = self.criterion(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1),
                                  torch.Tensor([1]).to(device=self.params.device))

            # if self.params.ont:
            #     g, _, _ = data_pos
            #     ent2types = np.array(self.ent2types)
            #     # head_ids = np.array([triplets[i][0] for i in range(len(triplets))], dtype=np.int)
            #     # tail_ids = np.array([triplets[i][1] for i in range(len(triplets))], dtype=np.int)
            #     head_ids = np.array((g.ndata['id'] == 1).nonzero().squeeze(1).cpu())
            #     tail_ids = np.array((g.ndata['id'] == 2).nonzero().squeeze(1).cpu())
            #     head_eids = g.ndata['eid'][head_ids].cpu()
            #     tail_eids = g.ndata['eid'][tail_ids].cpu()
            #     tmp = []
            #     if len(head_eids) == 1:
            #         tmp += [ent2types[head_eids], ent2types[tail_eids]]
            #     else:
            #         tmp += ent2types[head_eids].tolist()
            #         tmp += ent2types[tail_eids].tolist()
            #
            #     active_types = set(
            #         [tmp[i][j] for i in range(len(tmp)) for j in range(len(tmp[i])) for _ in range(len(tmp[i]))])
            #
            #     active_class_hier = []
            #     for (c, p) in self.class_hier:
            #         if c in active_types or p in active_types:
            #             active_class_hier.append([c, p])
            #     active_class_hier = torch.tensor(active_class_hier).to(self.params.device)
            #     active_class_hier = torch.tensor(active_class_hier).to(self.params.device)
            #     # print(active_class_hier)
            #     loss3 = 0.05 * torch.sqrt(torch.pow(
            #         self.ont_scorer.type_emb(active_class_hier[:, 0]) - self.ont_scorer.type_emb(
            #             active_class_hier[:, 1]) / active_class_hier.size(0),
            #         2))
            #     loss += loss3.sum()

            # loss = loss + loss_cls
            # print(score_pos, score_neg, loss)
            loss.backward()
            if self.params.ont:
                torch.nn.utils.clip_grad_norm(parameters=self.ont_scorer.parameters(), max_norm=0.5, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                if len(score_pos.squeeze().detach().cpu().size()) == 0:
                    continue
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss
                # total_cls_loss += loss_cls

            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                print('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc_roc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc_roc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        print(
                            f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc_roc']

        auc_roc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        # return total_loss, total_cls_loss, auc_roc, auc_pr, weight_norm

        return total_loss, auc_roc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            self.params.epoch = epoch
            # print(self.params.epoch)
            # time_start = time.time()
            # loss, cls_loss, auc_roc, auc_pr, weight_norm = self.train_epoch()
            loss, auc_roc, auc_pr, weight_norm = self.train_epoch()

            # time_elapsed = time.time() - time_start
            print(
                f'Epoch {epoch} with loss: {loss}, training auc_roc: {auc_roc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}')
            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir,
                                                       'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        if self.params.ont:
            torch.save(self.ont_scorer, os.path.join(self.params.exp_dir, 'best_ont_scorer.pth'))
        if self.params.type_graph:
            torch.save(self.type_scorer.state_dict(), os.path.join(self.params.exp_dir, 'best_type_scorer.pth'))
        print('Better models found w.r.t accuracy. Saved it!')
