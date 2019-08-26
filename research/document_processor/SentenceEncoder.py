import sys

import torch
from pytorch_transformers import *

from research.document_processor import PrepareInputForSentenceEncoder
from research.libnlp.Document import Document


class SentenceEncoder:
    MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased'),
              (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
              (GPT2Model, GPT2Tokenizer, 'gpt2'),
              (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
              (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
              (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024')]

    def __init__(self, model_type, task_type, logger, max_len):
        self.model_type = model_type  # indicates which sentence encoder you want to use
        self.task_type = task_type  # indicates which type of classification task you are performing
        self.max_len = max_len
        model, tokenizer, pretrained_weights = self.validate_model(model_type, logger)
        self.model = model.from_pretrained(pretrained_weights)
        self.tokenizer = tokenizer.from_pretrained(pretrained_weights)
        self.logger = logger

    def validate_model(self, type, logger):
        if type > len(self.MODELS):
            logger.error("Incorrect model-tokenizer-pretrained_weights combination")
            sys.exit()
        else:
            return self.MODELS[type]

    def get_embedding(self, d, take_hypernyms):
        input_ids, position_vect1, position_vect2 = PrepareInputForSentenceEncoder.convert_to_input(d, self.model_type, self.task_type, self.tokenizer,
                                                                    self.max_len, take_hypernyms, add_positional_features=True)
        with torch.no_grad():
            d.linkTokenIDs([input_ids, position_vect1, position_vect2])
            self.logger.info("Document " + str(d.doc_id) + " encoded ")
        return d

    def encode_text(self, document, take_hypernyms = True):
        if isinstance(document, Document):
            document = self.get_embedding(document, take_hypernyms)
        if isinstance(document, dict):
            for idx in document.keys():
                document[idx] = self.get_embedding(document[idx], take_hypernyms)
        return document
