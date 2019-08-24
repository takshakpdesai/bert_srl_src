class Document:
    def __init__(self, text, doc_id):
        self.tokens = list()
        self.doc_id = doc_id
        self.text = text
        self.sr = None
        self.input_features = None

    def linkTextToDoc(self, text):
        self.text = text

    def linkSemanticRelation(self, sr):
        self.sr = sr

    def linkTokenIDs(self, input_features):
        self.input_features = input_features
