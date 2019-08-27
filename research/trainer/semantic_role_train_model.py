import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import trange

from research.libnlp.Document import Document
from research.models.BERTModels import BERTForRoleLabeling


def create_class_map(document):
    sr_dict = dict()
    if isinstance(document, dict):
        for d in document.values():
            sr = d.sr.labels
            for label in sr:
                if label not in sr_dict.keys():
                    sr_dict[label] = len(sr_dict)
    if isinstance(document, Document):
        sr = document.sr.labels
        for label in sr:
            if label not in sr_dict.keys():
                sr_dict[label] = len(sr_dict)
    return sr_dict


def train_model(document, batch_size, lm_model, device, lr, epochs, logger):
    train_inputs = list()
    train_segments = list()
    train_masks = list()
    train_relations = list()
    position_vect1 = list()
    sr_dict = create_class_map(document)
    if isinstance(document, dict):
        for d in document.values():
            labels = list()
            [ip, segment, mask], v, padded_labels = d.input_features
            train_inputs.append(ip)
            train_segments.append(segment)
            train_masks.append(mask)
            position_vect1.append(v)
            for label in padded_labels:
                labels.append(sr_dict[label])
            train_relations.append(labels)
    elif isinstance(document, Document):
        labels = list()
        [train_inputs, train_segments, train_masks], v, _ = document.input_features
        position_vect1.append(v)
        for label in document.sr.labels:
            labels.append(sr_dict[label])
        train_relations.append(labels)
    train_inputs = torch.tensor(train_inputs)
    train_segments = torch.tensor(train_segments)
    train_masks = torch.tensor(train_masks)
    train_relations = torch.tensor(train_relations)
    position_vect1 = torch.tensor(position_vect1)

    train_data = TensorDataset(train_inputs, train_segments, train_masks, train_relations, position_vect1)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    our_model = BERTForRoleLabeling(lm_model, len(sr_dict), logger)
    our_model = nn.DataParallel(our_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])  # TODO: make param
    our_model.to(device)
    logger.info(str(our_model))

    param_optimizer = list(our_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = \
        [{"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}]

    optimizer = Adam(optimizer_grouped_parameters, lr=lr)
    tr_t, tr_l, vl_l, vl_a, f1_s = [], [], [], [], []

    for _ in trange(epochs, desc="Epoch"):
        our_model.train()
        tr_loss = 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_data_loader):
            batch = tuple(item.to(device) for item in batch)
            bt_features, bt_segments, bt_masks, bt_relations, bt_pos = batch
            loss = our_model(bt_features, bt_segments, bt_masks, position_vector = bt_pos, relation_labels=bt_relations)
            loss.sum().backward()
            tr_loss += loss.sum().item()
            nb_tr_examples += bt_features.size(0)
            nb_tr_steps += 1
            optimizer.step()
            our_model.zero_grad()
        tr_t.append(tr_loss)
        tr_l.append(tr_loss / nb_tr_steps)
        logger.info("Total training loss: {}".format(tr_loss))
        logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))

    return our_model, sr_dict
