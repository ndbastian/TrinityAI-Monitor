import math

import ipdb
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer
from pprint import pprint

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class Trainer(object):
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])
        pprint(opt)

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])


    def compute_logits(self, graphs):
        normalized_edge_weight = None
        device = graphs.device
        geo_feat_len = self.opt['geo_feat_len']
        #feat_option = self.opt['feat_option']
        # if feat_option == 'vis_feat':
        logits = self.model(graphs, graphs.ndata['feature'][:, geo_feat_len:].float().to(device),
                                normalized_edge_weight)

        return logits

    def update(self, graphs, target, idx):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.compute_logits(graphs)
        logits = torch.log_softmax(logits, dim=-1)
        # loss = self.criterion(logits[idx], target[idx])
        loss = self.criterion(logits, target)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, graphs, target, idx):
        """soft update"""
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.compute_logits(graphs)
        logits = torch.log_softmax(logits, dim=-1)
        #mulitply with the logits
        # loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        loss = -torch.mean(torch.sum(target * logits, dim=-1))

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, graphs, target, idx):
        self.model.eval()
        with torch.no_grad():
            logits = self.compute_logits(graphs)
            logits = torch.log_softmax(logits, dim=-1)
            target_idx = torch.max(target, dim=-1)[1]
            loss = self.criterion(logits, target_idx)
            # loss = self.criterion(logits[idx], target[idx])
            # preds = torch.max(logits[idx], dim=1)[1]
            # correct = preds.eq(target[idx]).double()
            preds = torch.max(logits, dim=1)[1]
            correct = preds.eq(target_idx).double()
            accuracy = correct.sum() / len(preds)

        return loss.item(), preds, accuracy.item()

    def predict(self, graphs, tau=1):
        self.model.eval()
        with torch.no_grad():
            logits = self.compute_logits(graphs) / tau
            logits = torch.log_softmax(logits, dim=-1)
        return logits

    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optim'])
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
