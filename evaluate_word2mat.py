"""
This script is for assessing the performance of models (or their subparts) AFTER training.
"""

from __future__ import absolute_import, division, unicode_literals

import sys, os, time
import torch
import logging
import pickle
import numpy as np
import pandas as pd
import argparse

from wrap_evaluation import run_and_evaluate

from data import get_index_batch

from torch.autograd import Variable

# Set PATHs
PATH_SENTEVAL = '/data22/fmai/data/SentEval/SentEval/'
PATH_TO_DATA = '/data22/fmai/data/SentEval/SentEval/data'

assert os.path.exists(PATH_SENTEVAL) and os.path.exists(PATH_TO_DATA), "Set path to SentEval + data correctly!"

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

total_time_encoding = 0.
total_samples_encoded = 0

if __name__ == "__main__":

    def prepare(params_senteval, samples):
    
        params = params_senteval["cmd_params"]

        # Load vocabulary
        vocabulary = pickle.load(open(params.word_vocab, "rb" ))[0]

        params_senteval['vocabulary'] = vocabulary
        params_senteval['inverse_vocab'] = {vocabulary[w] : w for w in vocabulary}
        params_senteval['encoders'] = [torch.load(p) for p in params.encoders]

    def _batcher_helper(encoder, vocabulary, batch):
        sent, _ = get_index_batch(batch, vocabulary)
        sent_cuda = Variable(sent.cuda())
        sent_cuda = sent_cuda.t()
        encoder.eval() # Deactivate drop-out and such
        embeddings = encoder.forward(sent_cuda).data.cpu().numpy()

        return embeddings

    def get_params_parser():
        parser = argparse.ArgumentParser(description='Evaluates the performance of a given encoder. Use --included_features to\
                                        evaluate subparts of the model.')

        # paths
        parser.add_argument('--word_vocab', type=str, default=None, help= \
            "Specify path where to load precomputed word.", required = True)
        parser.add_argument('--encoders', type=str, nargs='+', default=None, help= \
            "Specify path to load encoder models from.", required = True)
        parser.add_argument('--aggregation', type=str, default="concat", help= \
            "Specify operation to use for aggregating embeddings from multiple encoders.", required = False,
            choices = ['concat', 'add'])
        parser.add_argument('--gpu_device', type=int, default=0, help= \
            "You need to specify the id of the gpu that you used for training to avoid errors.", required = False)
        parser.add_argument('--add_start_end_token', action = "store_true", default=False, help= \
            "If activated, the start and end tokens are added to every sample. Used e.g. for NLI trained encoder.", required = False)
        parser.add_argument('--included_features', type = int, nargs = '+', default=None, help= \
            "If specified, expects two integers a and b, which denote the range (a is inclusive, b is exclusive) of indices to use from\
            the embedding. E.g., if '--included_features 0 300' is specified, the embedding that is evaluated consists only of the first\
            300 dimensions of the actual embedding: embeddings[a,b].", required = False)
        return parser

    def batcher(params_senteval, batch):

        start_time = time.time()
        params = params_senteval["cmd_params"]
        if params.add_start_end_token:
            batch = [['<s>'] + s + ['</s>'] for s in batch]

        embeddings_list = [_batcher_helper(enc, params_senteval['vocabulary'], batch) for enc in params_senteval['encoders']]

        if params.aggregation == "add":
            embeddings = sum(embeddings_list)

        elif params.aggregation == "concat":
            embeddings = np.hstack(embeddings_list)

        global total_time_encoding
        global total_samples_encoded
        total_time_encoding += time.time() - start_time
        total_samples_encoded += len(batch)

        if params.included_features:
            a = params.included_features[0]
            b = params.included_features[1]
            embeddings = embeddings[:, a:b]

        return embeddings

    def _load_encoder_and_eval(params):
        encoder_for_wordemb_eval = torch.load(params.encoders[0])
        return (encoder_for_wordemb_eval, [])
    run_and_evaluate(_load_encoder_and_eval, get_params_parser, batcher, prepare)

    print("Encoding speed: {}/s".format(total_samples_encoded / total_time_encoding))

