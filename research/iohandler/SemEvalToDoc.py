import re
import sys

from research.libnlp import Document, SemanticRelation


class SemEvalReader:
    TRAIN = 0
    TEST = 1

    def __init__(self, path, logger, mode):
        self.path = path
        self.mode = mode
        self.logger = logger
        self.documents = dict()


    def get_formatted_text(self, text, replace_by_mask = False, predictor = None):
        text = text[1:-1]  # removing double quotes from the beginning and end of sentence
        e1 = re.search("<e1>(.*?)</e1>", text)
        e1_text = e1.group().replace("<e1>", "").replace("</e1>", "")
        text = replace_by_entity_mask(text, e1_text, 1, replace_by_mask = replace_by_mask, predictor= predictor)
        e2 = re.search("<e2>(.*?)</e2>", text)
        e2_text = e2.group().replace("<e2>", "").replace("</e2>", "")
        text = replace_by_entity_mask(text, e2_text, 2, replace_by_mask = replace_by_mask, predictor= predictor)
        return text, e1_text, e2_text, e1.start(), e2.start()

    def read_file(self, predictor = None):
        try:
            reader_object = open(self.path)
            try:
                for sample in reader_object.read().split("\n\n"):
                    lines = sample.split("\n")
                    doc_id, dtext = lines[0].split("\t")
                    document_text, token1, token2, token1_offset, token2_offset = self.get_formatted_text(dtext, replace_by_mask=True, predictor=predictor)
                    d = Document.Document(document_text, int(doc_id))
                    relation_name, relation_direction = getSemanticRelation(lines[1], self.logger)
                    sr = SemanticRelation.SemanticRelation(token1, token2, relation_name, relation_direction)
                    d.linkSemanticRelation(sr)
                    self.documents[doc_id] = d
            except ValueError:
                self.logger.info("Reached end of file")
            reader_object.close()
            return self.documents
        except IOError:
            self.logger.error("File not found at path " + self.path)
            sys.exit("File not found at path " + self.path)


def getSemanticRelation(text, logger):
    relation_name = text.split("(")[0]
    if "(e1,e2)" in text:
        relation_direction = 0
    elif "(e2,e1)" in text:
        relation_direction = 1
    else:
        relation_direction = SemanticRelation.SemanticRelation.NO_DIRECTION
    logger.info("Relation " + relation_name + " found with directionality " + str(relation_direction))
    return relation_name, relation_direction

def replace_by_entity_mask(text, ent_text, ent_mention, replace_by_mask = False, predictor = None):
    if ent_mention == 1:
        e = "<e1>"
        e_ = "</e1>"
        pattern = r"<e1>(.*?)</e1>"
    else:
        e = r"<e2>"
        e_ = r"</e2>"
        pattern = r"<e2>(.*?)</e2>"
    if replace_by_mask is False:
        text = text.replace(e, " ").replace(e_, " ")
    else:
        if predictor is None:
            text = re.sub(pattern, " [MASK] ", text)
        else:
            ent_tokens = ent_text.split()
            doc = predictor(text)
            ent = " [ O ] "
            for token in doc.ents: # TODO: Fix based on offset values
                if token.text == ent_tokens[0]:
                    ent = " [ " + token.label_ + " ] "
            text = re.sub(pattern, ent, text)
    return text
