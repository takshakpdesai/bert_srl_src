import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

from research.evaluation.semeval2010_writer import file_writer
from research.libnlp.Document import Document


def test_semantic_relation_model(document, sr_dict, batch_size, model, device, true_file, prediction_file, logger):
    test_inputs = list()
    test_segments = list()
    test_masks = list()
    test_relations = list()
    test_directions = list()
    test_doc_ids = list()
    position_vect1 = list()
    position_vect2 = list()
    if isinstance(document, dict):
        for d in document.values():
            [ip, segment, mask], v1, v2 = d.input_features
            test_inputs.append(ip)
            test_segments.append(segment)
            test_masks.append(mask)
            test_relations.append(sr_dict[d.sr.sr])
            test_directions.append(d.sr.dir)
            test_doc_ids.append(d.doc_id)
            position_vect1.append(v1)
            position_vect2.append(v2)
    elif isinstance(document, Document):
        test_inputs, test_segments, test_masks = document.input_features
        test_relations.append(sr_dict[document.sr.sr])
        test_directions.append(document.sr.dir)
        test_doc_ids.append(document.doc_id)
    test_inputs = torch.tensor(test_inputs)
    test_segments = torch.tensor(test_segments)
    test_masks = torch.tensor(test_masks)
    test_relations = torch.tensor(test_relations)
    test_directions = torch.tensor(test_directions)
    test_doc_ids = torch.tensor(test_doc_ids)
    position_vect1 = torch.tensor(position_vect1)
    position_vect2 = torch.tensor(position_vect2)

    test_data = TensorDataset(test_inputs, test_segments, test_masks, test_relations, test_directions, test_doc_ids, position_vect1, position_vect2)
    test_sampler = RandomSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return get_relation_predictions(test_data_loader, model, device, sr_dict, true_file, prediction_file, logger)


def get_relation_predictions(test_data_loader, model, device, sr_dict, true_file, prediction_file, logger, num_directions=2):
    eval_loss = 0.0
    nb_eval_steps = 0
    inverse_sr_dict = {v: k for k, v in sr_dict.items()}
    file1 = open(true_file, "w+")
    file2 = open(prediction_file, "w+")
    model.eval()
    for batch in test_data_loader:
        batch = tuple(t.to(device) for t in batch)
        bt_features, bt_segments, bt_masks, bt_relations, bt_directions, bt_ids, bt_pos1, bt_pos2 = batch
        with torch.no_grad():
            loss = model(bt_features, bt_segments, bt_masks, position_vector1 = bt_pos1, position_vector2 = bt_pos2, relation_labels=bt_relations,
                         direction_labels=bt_directions)
            predicted_relations, predicted_directions = model(bt_features, bt_segments, bt_masks, position_vector1 = bt_pos1, position_vector2 = bt_pos2)
            predicted_relations = predicted_relations.view(-1, len(sr_dict.keys()))
            predicted_directions = predicted_directions.view(-1, num_directions)
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
        file1 = file_writer(bt_relations, bt_directions, bt_ids, file1, inverse_sr_dict)
        file2 = file_writer(predicted_relations, predicted_directions, bt_ids, file2, inverse_sr_dict,
                            class_type="predicted")
    logger.info("Total test loss: {}".format(eval_loss))
    logger.info("Test loss: {}".format(eval_loss / nb_eval_steps))

    file1.close()
    file2.close()

def get_role_predictions(document, test_data_loader, model, device, sr_dict, true_file, prediction_file, logger):
    eval_loss = 0.0
    nb_eval_steps = 0
    inverse_sr_dict = {v: k for k, v in sr_dict.items()}
    file1 = open(true_file, "w+")
    file2 = open(prediction_file, "w+")
    model.eval()
    for batch in test_data_loader:
        batch = tuple(t.to(device) for t in batch)
        bt_features, bt_segments, bt_masks, bt_relations, bt_ids, bt_pos = batch
        with torch.no_grad():
            loss = model(bt_features, bt_segments, bt_masks, position_vector = bt_pos, relation_labels=bt_relations)
            predicted_relations = model(bt_features, bt_segments, bt_masks, position_vector = bt_pos)

        eval_loss += loss.mean().item()
        nb_eval_steps += 1
        file1 = file_writer(document, bt_relations, bt_ids, file1, inverse_sr_dict)
        file2 = file_writer(document, predicted_relations, bt_ids, file2, inverse_sr_dict, class_type="predicted")
    logger.info("Total test loss: {}".format(eval_loss))
    logger.info("Test loss: {}".format(eval_loss / nb_eval_steps))

    file1.close()
    file2.close()


def test_semantic_role_model(document, sr_dict, batch_size, model, device, true_file, prediction_file, logger):
    test_inputs = list()
    test_segments = list()
    test_masks = list()
    test_relations = list()
    test_doc_ids = list()
    position_vect = list()
    if isinstance(document, dict):
        for d in document.values():
            tags = list()
            [ip, segment, mask], pos, labels = d.input_features
            test_inputs.append(ip)
            test_segments.append(segment)
            test_masks.append(mask)
            for label in labels:
                if label in sr_dict:
                    tags.append(sr_dict[label])
                else:
                    tags.append(sr_dict["O"]) # TODO: Need a better fix for this!!
            test_relations.append(tags)
            test_doc_ids.append(d.doc_id)
            position_vect.append(pos)
    test_inputs = torch.tensor(test_inputs)
    test_segments = torch.tensor(test_segments)
    test_masks = torch.tensor(test_masks)
    test_relations = torch.tensor(test_relations)
    test_doc_ids = torch.tensor(test_doc_ids)
    position_vect = torch.tensor(position_vect)

    test_data = TensorDataset(test_inputs, test_segments, test_masks, test_relations, test_doc_ids, position_vect)
    test_sampler = RandomSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return get_role_predictions(document, test_data_loader, model, device, sr_dict, true_file, prediction_file, logger)
