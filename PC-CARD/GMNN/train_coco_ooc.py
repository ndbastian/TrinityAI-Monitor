import sys
import os
import copy
from datetime import datetime
import time
sys.path.append(os.path.abspath("."))
import ipdb
import numpy as np
import random
from pprint import pprint
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import Trainer
from torch.utils.data import DataLoader
from data.data_loader import *
from models.registry import GCN_4_layer_fc

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, help='Data feature root directory')
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='gmnn_models')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1,
                    help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max',
                    help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--testrun', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

opt = vars(args)
##
opt['dataset'] = '../data/pubmed'
opt['hidden_dim'] = 16
opt['input_dropout'] = 0.5
opt['dropout'] = 0
# opt['optimizer'] = 'adam'
# opt['lr'] = 0.01
# opt['decay'] = 5e-4
opt['optimizer'] = 'adamw'
opt['lr'] = 0.001
opt['decay'] = 0
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 5 #pre-training
opt['epoch'] = 10 #each epoch
opt['iter'] = 1
opt['use_gold'] = 1
opt['draw'] = 'max'
opt['tau'] = 0.1
opt['num_class'] = 80
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#make sure output dir exists
os.makedirs(opt['save'], exist_ok=True)

num_classes = 80
h1_dim = 256
h2_dim = 128
h3_dim = 64
h4_dim = 64
in_feats = 2048
geo_feat_len = 7
##add geo_feat_len i.e geo_feats here
p_in_feats = num_classes + geo_feat_len
gnnp = GCN_4_layer_fc(p_in_feats, h1_dim, h2_dim, h3_dim, h4_dim, num_classes).to(device)
print(gnnp)
gnnq = GCN_4_layer_fc(in_feats, h1_dim, h2_dim, h3_dim, h4_dim, num_classes).to(device)
print(gnnq)

opt['geo_feat_len'] = geo_feat_len #only vis feat is used
opt['name'] = 'trainer_q'
trainer_q = Trainer(opt, gnnq)
#copy opt
opt_p = copy.deepcopy(opt)
opt_p['name'] = 'trainer_p'
opt_p['geo_feat_len'] = 0 #so that all features is used
trainer_p = Trainer(opt_p, gnnp)

train_batch_size = 1024
val_batch_size = 1024

data_root = args.data_root
graph_dir_train = os.path.join(data_root, "train")
graph_dir_test = os.path.join(data_root, "val")

# ----------------------- data loaders -----------------------------------------
print("curating graphs .....")
node_num_thresh = 4

if args.testrun:  # run test on few files
    graph_path_list_train_sel = sorted(glob(graph_dir_train + '/*.bin'))[:100]
    graph_path_list_test_sel = sorted(glob(graph_dir_test + '/*.bin'))[:100]
else:
    graph_path_list_train_sel = filter_graphs(graph_dir_train, node_num_thresh=node_num_thresh)
    graph_path_list_test_sel = filter_graphs(graph_dir_test, node_num_thresh=node_num_thresh)

random.seed(0)
val_sample_frac = 0.20  # 20% of the test samples
nval = int(len(graph_path_list_test_sel) * val_sample_frac)
graph_path_list_val_sel = random.sample(graph_path_list_test_sel, nval)

train_dataset = get_dataset(path_list=graph_path_list_train_sel)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

val_dataset = get_dataset(path_list=graph_path_list_val_sel)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size)

test_dataset = get_dataset(path_list=graph_path_list_test_sel)
test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size)
print("dataloaders prepared .....")

def quick_batch(batch_paths):
    batch, path_list = [], []
    for G_path in batch_paths:
        G, _ = load_graphs(G_path)
        G = G[0]
        batch.append(G)
        path_list.append(G_path)
    batch = dgl.batch(batch).to(device)
    return path_list, batch


def update_p_data(inputs_q , target_p):
    preds = trainer_q.predict(inputs_q, opt['tau'])
    target_p_orig = target_p.clone()
    if opt['draw'] == 'exp':
        target_p = preds.clone()
    #TODO: lets say we randomize here by 50% or 75%
    # or just take target_p and randomize gold by 50% and see
    elif opt['draw'] == 'max':
        idx_lb = torch.max(preds, dim=-1)[1]
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    elif opt['draw'] == 'smp':
        #TODO: make sure the multinomail gets only positve values
        positive_matrix = (preds - preds.min()) / (preds.max() - preds.min())
        # idx_lb = torch.multinomial(preds, 1).squeeze(1)
        idx_lb = torch.multinomial(positive_matrix, 1).squeeze(1)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    if opt['use_gold'] == 1:
        #TODO: return true label here
        target_p = target_p_orig
    inputs_p = target_p.clone()
    return inputs_p, target_p

def update_q_data(inputs_p, target_q):
    preds = trainer_p.predict(inputs_p)
    if opt['use_gold'] == 1:
        #TODO: return true label here
        return target_q
    else:
        return preds.clone()


def validate(dataloader):
    all_graph_predictions = []
    all_graph_test_labels = []

    for step, G_batch in tqdm(enumerate(dataloader), desc="Validating", total=len(dataloader)):
        G_path_list, batch = quick_batch(G_batch)
        # TODO: fix this: label is subtracted -1 since 1 to 80
        per_graph_labels = batch.ndata['label'].to(device) - 1
        target_onehot = torch.zeros(len(per_graph_labels), opt['num_class']).to(device)
        target_onehot.scatter_(1, per_graph_labels.unsqueeze(1), 1.0)
        _, pred_labels, accuracy_dev = trainer_q.evaluate(batch, target_onehot, None)
        all_graph_predictions.append(pred_labels.cpu())
        all_graph_test_labels.append(per_graph_labels.to('cpu'))

    all_node_predictions = np.array(
        [node_pred.item() for graph_pred in all_graph_predictions for node_pred in graph_pred])
    all_node_test_labels = np.array(
        [node_labels.item() for graph_labels in all_graph_test_labels for node_labels in graph_labels])
    accuracy = (all_node_predictions == all_node_test_labels).sum() / len(all_node_test_labels)
    return accuracy


def validate_p(dataloader):
    all_graph_predictions = []
    all_graph_test_labels = []

    for step, G_batch in tqdm(enumerate(dataloader), desc="Validating", total=len(dataloader)):
        G_path_list, batch = quick_batch(G_batch)
        # TODO: fix this: label is subtracted -1 since 1 to 80
        per_graph_labels = batch.ndata['label'].to(device) - 1
        target_onehot = torch.zeros(len(per_graph_labels), opt['num_class']).to(device)
        target_onehot.scatter_(1, per_graph_labels.unsqueeze(1), 1.0)
        inputs_p, target_p = update_p_data(batch, target_onehot)  # batch = num_nodes x num_features
        # batch.ndata['feature'] = inputs_p # num_nodes x num_class
        # ----------------
        inputs_p_new = torch.cat((inputs_p, batch.ndata['geo_feature']), axis=1)
        batch.ndata['feature'] = inputs_p_new  # num_nodes x num_class
        # ----------------
        _, pred_labels, accuracy = trainer_p.evaluate(batch, target_onehot, None)
        all_graph_predictions.append(pred_labels.cpu())
        all_graph_test_labels.append(per_graph_labels.to('cpu'))

    all_node_predictions = np.array(
        [node_pred.item() for graph_pred in all_graph_predictions for node_pred in graph_pred])
    all_node_test_labels = np.array(
        [node_labels.item() for graph_labels in all_graph_test_labels for node_labels in graph_labels])
    accuracy = (all_node_predictions == all_node_test_labels).sum() / len(all_node_test_labels)
    return accuracy


def pre_train_q(epoches):
    best = 0.0
    results = []
    idx_train = None
    for epoch in range(epoches):
        for step, G_batch in tqdm(enumerate(train_dataloader), desc="Training  ", total=len(train_dataloader)):
                G_path_list , batch = quick_batch(G_batch)
                # TODO: fix this: label is subtracted -1 since 1 to 80
                per_graph_labels = batch.ndata['label'].to(device) - 1
                target_onehot = torch.zeros(len(per_graph_labels), opt['num_class']).to(device)
                target_onehot.scatter_(1, per_graph_labels.unsqueeze(1), 1.0)
                loss = trainer_q.update_soft(batch, target_onehot, idx_train)

        accuracy_dev = validate(val_dataloader)
        accuracy_test = 0
        # accuracy_test = validate(test_dataloader)
        results += [(accuracy_dev, accuracy_test)]
        if accuracy_dev > best:
            best = accuracy_dev
            accuracy_test = validate(test_dataloader) #only evaluate then ... #TODO: fix this later
            results[-1] = (accuracy_dev, accuracy_test) #replace here
            state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())),
                          ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
    trainer_q.model.load_state_dict(state['model'])
    trainer_q.optimizer.load_state_dict(state['optim'])
    return results

#TODO: aggreagate accuracy and losses after each batch
def train_p(epoches):
    results = []
    idx_all = None
    for epoch in range(epoches):
        for step, G_batch in tqdm(enumerate(train_dataloader), desc="P Training  ", total=len(train_dataloader)):
                G_path_list , batch = quick_batch(G_batch)
                # TODO: fix this: label is subtracted -1 since 1 to 80
                per_graph_labels = batch.ndata['label'].to(device) - 1
                target_onehot = torch.zeros(len(per_graph_labels), opt['num_class']).to(device)
                target_onehot.scatter_(1, per_graph_labels.unsqueeze(1), 1.0)
                inputs_p, target_p = update_p_data(batch, target_onehot) # batch = num_nodes x num_features
                # batch.ndata['feature'] = inputs_p # num_nodes x num_class
                #----------------
                inputs_p_new = torch.cat((inputs_p, batch.ndata['geo_feature']), axis=1)
                batch.ndata['feature'] = inputs_p_new # num_nodes x num_class
                #----------------
                loss = trainer_p.update_soft(batch, target_p, idx_all)

        accuracy_dev = validate_p(val_dataloader)
        accuracy_test = validate_p(test_dataloader)
        results += [(accuracy_dev, accuracy_test)]
    return results

def train_q(epoches):
    results = []
    idx_all = None
    for epoch in range(epoches):
        for step, G_batch in tqdm(enumerate(train_dataloader), desc="Q Training  ", total=len(train_dataloader)):
                G_path_list , batch = quick_batch(G_batch)
                # TODO: fix this: label is subtracted -1 since 1 to 80
                per_graph_labels = batch.ndata['label'].to(device) - 1
                target_onehot = torch.zeros(len(per_graph_labels), opt['num_class']).to(device)
                target_onehot.scatter_(1, per_graph_labels.unsqueeze(1), 1.0)
                #https://docs.dgl.ai/en/0.6.x/generated/dgl.DGLGraph.to.html#dgl.DGLGraph.to
                #copy the graphs to same device
                #save old feature to run throught network p which takes num_nodes x num_class
                feats_old = batch.ndata['feature'].clone().detach()
                inputs_p = batch
                # inputs_p.ndata['feature'] = target_onehot # num_nodes x num_class
                # ----------------
                inputs_p_new = torch.cat((target_onehot, batch.ndata['geo_feature']), axis=1)
                inputs_p.ndata['feature'] = inputs_p_new # num_nodes x num_class
                # ----------------
                target_q = update_q_data(inputs_p, target_onehot) # num_nodes x num_class
                #copy back to run throught network q which takes num_nodes x num_features
                batch.ndata['feature'] = feats_old
                loss = trainer_q.update_soft(batch, target_q, idx_all)

        accuracy_dev = validate(val_dataloader)
        accuracy_test = validate(test_dataloader)
        results += [(accuracy_dev, accuracy_test)]
    return results

def get_accuracy(results):
    best_dev, acc_test = 0.0, 0.0
    for d, t in results:
        if d > best_dev:
            best_dev, acc_test = d, t
    return acc_test

base_results, q_results, p_results = [], [], []
base_results += pre_train_q(opt['pre_epoch'])
pretraining_acc = get_accuracy(base_results)
print('Pretraining acc: {:.2f}%'.format(pretraining_acc * 100))
print("Pre-training done....")
print("------------------------")
for k in range(opt['iter']):
    p_results += train_p(opt['epoch'])
    q_results += train_q(opt['epoch'])

acc_test = get_accuracy(q_results)
print('Final Q-acc: {:.2f}%'.format(acc_test * 100))
acc_test_p = get_accuracy(p_results)
print('Final P-acc: {:.2f}%'.format(acc_test_p * 100))
if opt['save'] != '/':
    trainer_q.save(os.path.join(opt['save'],'gnnq.pt'))
    trainer_p.save(os.path.join(opt['save'],'gnnp.pt'))
