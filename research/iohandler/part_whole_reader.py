import sys

from research.libnlp import Document, SemanticRelation


class PartWholeReader:
    TRAIN = 0
    TEST = 1

    def __init__(self, path, logger, mode):
        self.path = path
        self.mode = mode
        self.logger = logger
        self.documents = dict()

    def read_file(self, predictor=None):
        try:
            reader_object = open(self.path)
            for doc_id, line in enumerate(reader_object.readlines()):
                token1 = line.split("\t")[0]
                token2 = line.split("\t")[2]
                sentence = line.split("\t")[6]
                relation_name = line.split("\t")[8]
                if predictor is not None:
                    sentence = replace_by_entity_mask(token1, sentence, predictor)
                    sentence = replace_by_entity_mask(token2, sentence, predictor)
                d = Document.Document(sentence, doc_id)
                sr = SemanticRelation.SemanticRelation(token1, token2, relation_name, SemanticRelation.SemanticRelation.NO_DIRECTION)
                d.linkSemanticRelation(sr)
                self.logger.info("Relation " + relation_name + " found!")
                self.documents[doc_id] = d
            reader_object.close()
            return self.documents
        except IOError:
            self.logger.error("File not found at path " + self.path)
            sys.exit("File not found at path " + self.path)

def replace_by_entity_mask(entity, sentence, predictor):
    if predictor is None:
        sentence = sentence.replace(entity, " [MASK] ")
    else:
        ent_tokens = sentence.split()
        doc = predictor(sentence)
        ent = " [ O ] "
        for token in doc.ents:  # TODO: Fix based on offset values
            if token.text == ent_tokens[0]:
                ent = " [ " + token.label_ + " ] "
        sentence = sentence.replace(entity, ent)
    return sentence
