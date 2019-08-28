import torch
import torch.nn as nn

from research.libnlp.SemanticRelation import SemanticRelation


def custom_loss_fct(loss_fct, predicted, labels, odd_one=SemanticRelation.NO_DIRECTION):
    true, pred = list(), list()
    for i, label in enumerate(labels):
        if label != odd_one:
            true.append(label)
            pred.append(predicted[i])
    try:
        return loss_fct(torch.stack(pred), torch.stack(true))
    except RuntimeError:  # if either list is empty
        return 0


class BERTForRelationClassification(nn.Module):
    def __init__(self, lm, num_relations, logger, num_directions=2):
        super(BERTForRelationClassification, self).__init__()
        self.language_model = lm
        self.relation_classifier = nn.Linear(770, num_relations)  # TODO: get from config, for small or large
        self.direction_classifier = nn.Linear(770, num_directions)
        self.num_relations = num_relations
        self.num_directions = num_directions
        self.logger = logger

    def forward(self, input_ids, input_segments, input_masks, position_vector1 = None, position_vector2 = None, relation_labels=None, direction_labels=None):
        pooled_output = self.language_model(input_ids, token_type_ids = input_segments, attention_mask = input_masks)[0]
        if position_vector1 is not None:
            pooled_output = torch.cat((pooled_output, position_vector1.float().unsqueeze(-1), position_vector2.float().unsqueeze(-1)), -1)
        pooled_output = pooled_output[:,-1]
        predicted_relations = self.relation_classifier(pooled_output)
        predicted_directions = self.direction_classifier(pooled_output)
        if relation_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            relation_loss = loss_fct(predicted_relations.view(-1, self.num_relations), relation_labels.view(-1))
            direction_loss = custom_loss_fct(loss_fct, predicted_directions, direction_labels)
            return relation_loss + direction_loss
        else:
            return predicted_relations, predicted_directions

class BERTForRoleLabeling(nn.Module):
    def __init__(self, lm, num_relations, logger):
        super(BERTForRoleLabeling, self).__init__()
        self.language_model = lm
        self.relation_classifier = nn.Linear(769, num_relations) # TODO: get from config, for small or large
        self.num_relations = num_relations
        self.logger = logger

    def forward(self, input_ids, input_segments, input_masks, position_vector = None, relation_labels=None):
        pooled_output = self.language_model(input_ids, token_type_ids = input_segments, attention_mask = input_masks)[0]
        if position_vector is not None:
            pooled_output = torch.cat((pooled_output, position_vector.float().unsqueeze(-1)), -1)
        predicted_relations = self.relation_classifier(pooled_output)
        if relation_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            relation_loss = loss_fct(predicted_relations.view(-1, self.num_relations), relation_labels.view(-1))
            return relation_loss
        else:
            return predicted_relations

class BERTForMultiTaskRelationClassification(nn.Module):
    def __init__(self, lm, num_orig_relations, num_pw_relations, logger, num_directions=2):
        super(BERTForMultiTaskRelationClassification, self).__init__()
        self.language_model = lm
        self.relation_classifier = nn.Linear(770, num_orig_relations)  # TODO: get from config, for small or large
        self.pw_classifier = nn.Linear(770, num_pw_relations)
        self.direction_classifier = nn.Linear(770, num_directions)
        self.num_orig_relations = num_orig_relations
        self.num_pw_relations = num_pw_relations
        self.num_directions = num_directions
        self.logger = logger

    def forward(self, input_ids, input_segments, input_masks, position_vector1 = None, position_vector2 = None, relation_labels=None, direction_labels=None, flag = False):
        pooled_output = self.language_model(input_ids, token_type_ids = input_segments, attention_mask = input_masks)[0]
        if position_vector1 is not None:
            pooled_output = torch.cat((pooled_output, position_vector1.float().unsqueeze(-1), position_vector2.float().unsqueeze(-1)), -1)
        pooled_output = pooled_output[:,-1]
        if flag:
            predicted_relations = self.pw_classifier(pooled_output)
            predicted_directions = None
        else:
            predicted_relations = self.relation_classifier(pooled_output)
            predicted_directions = self.direction_classifier(pooled_output)
        if relation_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if flag:
                relation_loss = loss_fct(predicted_relations.view(-1, self.num_pw_relations), relation_labels.view(-1))
                return relation_loss
            else:
                relation_loss = loss_fct(predicted_relations.view(-1, self.num_orig_relations), relation_labels.view(-1))
                direction_loss = custom_loss_fct(loss_fct, predicted_directions, direction_labels)
                return relation_loss + direction_loss
        else:
            return predicted_relations, predicted_directions