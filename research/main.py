import argparse
import logging
import subprocess

import torch
import spacy

from research.document_processor.SentenceEncoder import SentenceEncoder
from research.iohandler import SemEvalToDoc
from research.tester.test_model import test_model
from research.trainer.train_model import train_model

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-tr", "--path_to_training_file", help="Provide path to training file")
argument_parser.add_argument("-dev", "--path_to_dev_file", help="Provide path to dev file")
argument_parser.add_argument("-te", "--path_to_test_file", help="Provide path to test file")
argument_parser.add_argument("-log", "--path_to_log_file", help="Provide path to log file")
argument_parser.add_argument("-model", "--model_type", help="Provide model type")
argument_parser.add_argument("-max_len", "--max_len", help="Provide maximum sequence length")
argument_parser.add_argument("-b", "--batch_size", help="Provide batch size to work with")
argument_parser.add_argument("-lr", "--learning_rate", help="Provide learning rate to work with")
argument_parser.add_argument("-epochs", "--epochs", help="Number of training epochs")
argument_parser.add_argument("-model_path", "--model_path", help="Path where you want to save the model")
argument_parser.add_argument("-true_file", "--true_file", help="Path where you want to save the true relations")
argument_parser.add_argument("-prediction_file", "--prediction_file",
                             help="Path where you want to save the predictions")
argument_parser.add_argument("-perl", "--perl_eval_script", help="Path to the Perl evaluation script")

parse = argument_parser.parse_args()

# set up GPU
torch.manual_seed(0)
torch.cuda.set_device(0)
device = torch.device('cuda:0') # TODO: ensure default device is first item in list
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set up logging session
logging.basicConfig(filename=parse.path_to_log_file, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# declare objects:
nlp = spacy.load('en') # TODO: Make param
s = SentenceEncoder(int(parse.model_type), 0, logger, int(parse.max_len)) # TODO: Change to arg parse parameter
train_reader = SemEvalToDoc.SemEvalReader(parse.path_to_training_file, logger, SemEvalToDoc.SemEvalReader.TRAIN)
test_reader = SemEvalToDoc.SemEvalReader(parse.path_to_test_file, logger, SemEvalToDoc.SemEvalReader.TEST)

# go through pipeline:

training_docs = train_reader.read_file(spacy_model=nlp)
training_docs = s.encode_text(training_docs)
test_docs = test_reader.read_file(spacy_model=nlp)
test_docs = s.encode_text(test_docs)

model, sr_dict = train_model(training_docs, int(parse.batch_size), s.model, device, float(parse.learning_rate),
                             int(parse.epochs), logger)

test_model(test_docs, sr_dict, int(parse.batch_size), model, device, parse.true_file, parse.prediction_file, logger)
subprocess.call(["perl", parse.perl_eval_script, parse.true_file, parse.prediction_file])
