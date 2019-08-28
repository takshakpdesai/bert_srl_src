import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import trange

from research.libnlp.Document import Document
from research.models.BERTModels import BERTForRelationClassification, BERTForMultiTaskRelationClassification


def create_class_map(document):
    sr_dict = dict()
    if isinstance(document, dict):
        for d in document.values():
            sr = d.sr.sr
            if sr not in sr_dict.keys():
                sr_dict[sr] = len(sr_dict)
    if isinstance(document, Document):
        sr = document.sr.sr
        sr_dict[sr] = len(sr_dict)
    return sr_dict

def feature_extractor(document, flag = True):
    train_inputs = list()
    train_segments = list()
    train_masks = list()
    train_relations = list()
    train_directions = list()
    position_vect1 = list()
    position_vect2 = list()
    sr_dict = create_class_map(document)
    if isinstance(document, dict):
        for d in document.values():
            [ip, segment, mask], v1, v2 = d.input_features
            train_inputs.append(ip)
            train_segments.append(segment)
            train_masks.append(mask)
            position_vect1.append(v1)
            position_vect2.append(v2)
            train_relations.append(sr_dict[d.sr.sr])
            train_directions.append(d.sr.dir)
    elif isinstance(document, Document):
        train_inputs, train_segments, train_masks = document.input_features
        train_relations.append(sr_dict[document.sr.sr])
        train_directions.append(document.sr.dir)
    train_inputs = torch.tensor(train_inputs)
    train_segments = torch.tensor(train_segments)
    train_masks = torch.tensor(train_masks)
    train_relations = torch.tensor(train_relations)
    if flag:
        train_directions = torch.tensor(train_directions)
    else:
        train_directions = None
    position_vect1 = torch.tensor(position_vect1)
    position_vect2 = torch.tensor(position_vect2)

    return train_inputs, train_segments, train_masks, train_relations, train_directions, position_vect1, position_vect2, sr_dict

def train_model(document1, document2, batch_size, lm_model, device, lr, epochs, logger):
    train_inputs, train_segments, train_masks, train_relations, train_directions, position_vect1, position_vect2, sr_dict = feature_extractor(document1)
    pw_train_inputs, pw_train_segments, pw_train_masks, pw_train_relations, _, pw_position_vect1, pw_position_vect2, pw_sr_dict = feature_extractor(document2)

    train_data = TensorDataset(train_inputs, train_segments, train_masks, train_relations, train_directions, position_vect1, position_vect2)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    pw_train_data = TensorDataset(pw_train_inputs, pw_train_segments, pw_train_masks, pw_train_relations, pw_position_vect1, pw_position_vect2)
    pw_train_sampler = RandomSampler(pw_train_data)
    pw_train_data_loader = DataLoader(pw_train_data, sampler=pw_train_sampler, batch_size=batch_size)

    our_model = BERTForMultiTaskRelationClassification(lm_model, len(sr_dict), len(pw_sr_dict), logger)
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

    our_model.train()

    for _ in trange(epochs, desc="Epoch"):
        tr_loss = 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for _, batch in enumerate(pw_train_data_loader):
            batch = tuple(item.to(device) for item in batch)
            bt_features, bt_segments, bt_masks, bt_relations, bt_pos1, bt_pos2 = batch
            loss = our_model(bt_features, bt_segments, bt_masks, position_vector1 = bt_pos1, position_vector2 = bt_pos2, relation_labels=bt_relations, direction_labels=None, flag = True)
            loss.sum().backward()
            tr_loss += loss.sum().item()
            nb_tr_examples += bt_features.size(0)
            nb_tr_steps += 1
            optimizer.step()
            our_model.zero_grad()
        tr_t.append(tr_loss)
        tr_l.append(tr_loss / nb_tr_steps)
        logger.info("Total PW training loss: {}".format(tr_loss))
        logger.info("Train PW loss: {}".format(tr_loss / nb_tr_steps))

        tr_loss = 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for _, batch in enumerate(train_data_loader):
            batch = tuple(item.to(device) for item in batch)
            bt_features, bt_segments, bt_masks, bt_relations, bt_dirs, bt_pos1, bt_pos2 = batch
            loss = our_model(bt_features, bt_segments, bt_masks, position_vector1 = bt_pos1, position_vector2 = bt_pos2, relation_labels=bt_relations, direction_labels=bt_dirs)
            loss.sum().backward()
            tr_loss += loss.sum().item()
            nb_tr_examples += bt_features.size(0)
            nb_tr_steps += 1
            optimizer.step()
            our_model.zero_grad()
        tr_t.append(tr_loss)
        tr_l.append(tr_loss / nb_tr_steps)
        logger.info("Total SemEval training loss: {}".format(tr_loss))
        logger.info("Train SemEval loss: {}".format(tr_loss / nb_tr_steps))

    return our_model, sr_dict
