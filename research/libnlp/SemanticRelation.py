class SemanticRelation:

    NO_DIRECTION = 2 # indicates no directionality

    def __init__(self, token1, token2, sr, dir):
        self.token1 = token1
        self.token2 = token2
        self.sr = sr
        self.dir = dir
