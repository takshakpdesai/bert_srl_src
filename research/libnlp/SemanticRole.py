class SemanticRole:
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def get_verb(self):
        for i, label in enumerate(self.labels):
            if 'B-V' in label:
                verb_start = i
                verb_end = verb_start
            if 'I-V' in label:
                verb_end = i
        return ' '.join(self.tokens[verb_start:verb_end + 1])