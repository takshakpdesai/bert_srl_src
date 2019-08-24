class Token:
    def __init__(self, text, start, named_entity):
        self.text = text
        self.start = start
        self.ne = named_entity
