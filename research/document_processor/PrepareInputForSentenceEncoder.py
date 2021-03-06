from research.iohandler.SRLToDoc import realign_data

def extend_list(l, max_len):
    m = l
    m.extend([0] * (max_len - len(l)))
    return m

def extend_labels(l, max_len):
    m = l
    m.extend(['O'] * (max_len - len(l)))
    return m

def add_positional_features_to_text(ttext, token):
    position_vector = list()
    start_position = ttext.index(token[0])
    end_position = ttext.index(token[-1])
    for i, t in enumerate(ttext):
        if i < start_position:
            position_vector.append(start_position - i)
        if start_position <= i <= end_position:
            position_vector.append(0)
        if i > end_position:
            position_vector.append(i - end_position)
    return position_vector


def convert_to_input(document, type, task_type, tokenizer, max_len, add_positional_features = False):
    text = document.text
    if type == 0:  # for BERT
        if task_type == 0:  # TODO: for semantic relations
            position_vect1 = None
            position_vect2 = None
            token1 = document.sr.token1
            token2 = document.sr.token2
            ttext = "[CLS] " + text + " [SEP] " + token1 + " [SEP] " + token2
            tokenized_text = tokenizer.tokenize(ttext)
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            if add_positional_features:
                position_vect1 = add_positional_features_to_text(tokenized_text, tokenizer.tokenize(token1))
                position_vect2 = add_positional_features_to_text(tokenized_text, tokenizer.tokenize(token2))
                position_vect1 = extend_list(position_vect1, max_len)
                position_vect2 = extend_list(position_vect2, max_len)
            input_segments = [0] * (len(tokenizer.tokenize(text)) + 1) + [1] * (len(tokenizer.tokenize(token1)) + 1) + [
                        1] * (len(tokenizer.tokenize(token2)) + 1)
            input_masks = [1] * (len(tokenized_text))
            input_ids = extend_list(input_ids, max_len)
            input_segments = extend_list(input_segments, max_len)
            input_masks = extend_list(input_masks, max_len)
            return [input_ids, input_segments, input_masks], position_vect1, position_vect2

        if task_type == 1: # TODO: for semantic roles
            position_vect = None
            tokens = document.sr.tokens
            labels = document.sr.labels
            tokenized_tokens = tokenizer.tokenize(text)
            labels = realign_data(tokens, labels, tokenized_tokens)
            ttext = "[CLS] " + text + " [SEP] " + document.sr.get_verb()
            tokenized_text = tokenizer.tokenize(ttext)
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            if add_positional_features:
                position_vect = add_positional_features_to_text(tokenized_text, tokenizer.tokenize(document.sr.get_verb()))
                position_vect = extend_list(position_vect, max_len)
            input_segments = [0] * (len(tokenizer.tokenize(text)) + 1) + [1] * (len(tokenizer.tokenize(document.sr.get_verb())) + 1)
            input_masks = [1] * (len(tokenized_text))
            input_ids = extend_list(input_ids, max_len)
            input_segments = extend_list(input_segments, max_len)
            input_masks = extend_list(input_masks, max_len)
            labels = extend_labels(labels, max_len)
            return [input_ids, input_segments, input_masks], position_vect, labels

    return None