def get_prediction(relations):
    return relations

def file_writer(document, relations, doc_ids, file, inverse_sr_dict, class_type="true"):
    for doc_id in doc_ids:
        document_object = document[doc_id]
        tokens = document_object.sr.tokens
        if class_type == "true":
            labels = document_object.sr.labels
        else:
            labels = get_prediction(relations)

def conll_writer(tokens, labels):
    pass