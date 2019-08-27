import sys

from research.libnlp import Document, SemanticRole

class SRLReader:
    TRAIN = 0
    TEST = 1

    def __init__(self, path, logger, mode):
        self.path = path
        self.mode = mode
        self.logger = logger
        self.documents = dict()

    def read_file(self):
        try:
            reader_object = open(self.path)
            for doc_id, line in enumerate(reader_object.readlines()):
                text = ' '.join(line.split()[1:])
                doc_text = text.split("|||")[0]
                labels = text.split("|||")[1].split()
                d = Document.Document(doc_text, doc_id)
                s = SemanticRole.SemanticRole(doc_text.split(), labels)
                d.linkSemanticRelation(s)
                self.documents[doc_id] = d
            reader_object.close()
            return self.documents
        except IOError:
            sys.exit("File not found at "+self.path)

def realign_data(actual_tokens, class_list, bert_tokens):
    actual_tokens = [i.lower() for i in actual_tokens]
    new_class_list = list()
    i = 0
    j = 0
    while j < len(bert_tokens):
        a_token = actual_tokens[i]
        b_token = bert_tokens[j]
        token_c = class_list[i]
        if b_token == a_token:
            new_class_list.append(token_c)
            i += 1
            j += 1
        else:
            b_token = b_token.replace('#', '')
            while b_token in a_token:
                new_class_list.append(token_c)
                a_token = a_token.replace(b_token, '', 1)
                j += 1
                if j < len(bert_tokens):
                    b_token = bert_tokens[j].replace('#', '')
                else:
                    break
            i += 1
    return new_class_list