"""
Main script for training a Word2Mat model.
"""

import os
import sys
import time
import argparse
import pickle
import random

import numpy as np
from random import shuffle

import torch
from torch.autograd import Variable
import torch.nn as nn

# CBOW data
from torch.utils.data import DataLoader
from cbow import CBOWNet, build_vocab, tokenize, CBOWDataset, recursive_file_list, get_index_batch

# Encoder
from mutils import get_optimizer, run_hyperparameter_optimization, write_to_csv
from word2mat import get_cbow_cmow_hybrid_encoder, get_cbow_encoder, get_cmow_encoder
from torch.utils.data.sampler import SubsetRandomSampler

from wrap_evaluation import run_and_evaluate, construct_model_name

import time

def run_experiment(params):

    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    """
    DATA
    """
    dataset_path = params.dataset_path

    # build training and test corpus
    filename_list = recursive_file_list(dataset_path)
    print('Use the following files for training: ', filename_list)
    corpus = CBOWDataset(dataset_path, params.num_docs, params.context_size, 
                         params.num_samples_per_item, params.mode,
                         params.precomputed_word_vocab, params.max_words, 
                         None, 1000, params.precomputed_chunks_dir, params.temp_path)
    corpus_len = len(corpus)

    ## split train and test
    inds = list(range(corpus_len))
    shuffle(inds)

    num_val_samples = int(corpus_len * params.validation_fraction)
    train_indices = inds[:-num_val_samples] if num_val_samples > 0 else inds
    test_indices = inds[-num_val_samples:] if num_val_samples > 0 else []

    cbow_train_loader = DataLoader(corpus, sampler = SubsetRandomSampler(train_indices), batch_size=params.batch_size, shuffle=False, num_workers = params.num_workers, pin_memory = True, collate_fn = corpus.collate_fn)
    cbow_test_loader = DataLoader(corpus, sampler = SubsetRandomSampler(test_indices), batch_size=params.batch_size, shuffle=False, num_workers = params.num_workers, pin_memory = True, collate_fn = corpus.collate_fn)

    ## extract some variables needed for training
    num_training_samples = corpus.num_training_samples
    word_vec = corpus.word_vec
    unigram_dist = corpus.unigram_dist
    word_vec_copy = corpus._word_vec_count_tuple

    print("Number of sentences used for training:", str(num_training_samples))

    """
    MODEL
    """

    # build path where to store the encoder
    outputmodelname = construct_model_name(params.outputmodelname, params)

    # build encoder
    n_words = len(word_vec)
    if params.w2m_type == "cmow":
        encoder = get_cmow_encoder(n_words, padding_idx = 0, 
                                 word_emb_dim = params.word_emb_dim, 
                                 initialization_strategy = params.initialization)
        output_embedding_size = params.word_emb_dim
    elif params.w2m_type == "cbow":
        encoder = get_cbow_encoder(n_words, padding_idx = 0, word_emb_dim = params.word_emb_dim)
        output_embedding_size = params.word_emb_dim
    elif params.w2m_type == "hybrid":
        encoder = get_cbow_cmow_hybrid_encoder(n_words, padding_idx = 0,
                                 word_emb_dim = params.word_emb_dim,
                                 initialization_strategy = params.initialization)
        output_embedding_size = 2 * params.word_emb_dim

    # build cbow model
    cbow_net = CBOWNet(encoder, output_embedding_size, n_words, 
                       weights = unigram_dist, n_negs = params.n_negs, padding_idx = 0)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for training!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        cbow_net = nn.DataParallel(cbow_net)
        use_multiple_gpus = True
    else:
        use_multiple_gpus = False

    # optimizer
    print([x.size() for x in cbow_net.parameters()])
    optim_fn, optim_params = get_optimizer(params.optimizer)
    optimizer = optim_fn(cbow_net.parameters(), **optim_params)

    # cuda by default
    cbow_net.cuda()

    """
    TRAIN
    """
    val_acc_best = -1e10
    adam_stop = False
    stop_training = False
    lr = optim_params['lr'] if 'sgd' in params.optimizer else None

    # compute learning rate schedule
    if params.linear_decay:
        lr_shrinkage = (lr - params.minlr) / ((float(num_training_samples) / params.batch_size) * params.n_epochs)


    def forward_pass(X_batch, tgt_batch, params, check_size = False):

        X_batch = Variable(X_batch).cuda()
        tgt_batch = Variable(torch.LongTensor(tgt_batch)).cuda()
        k = X_batch.size(0)  # actual batch size

        loss = cbow_net(X_batch, tgt_batch).mean()
        return loss, k


    def validate(data_loader):
        cbow_net.eval()

        with torch.no_grad():
            all_costs = []
            for X_batch, tgt_batch in data_loader:
                loss, k = forward_pass(X_batch, tgt_batch, params)
                all_costs.append(loss.item())

        cbow_net.train()
        return np.mean(all_costs)

    def trainepoch(epoch):
        print('\nTRAINING : Epoch ' + str(epoch))
        cbow_net.train()
        all_costs = []
        logs = []
        words_count = 0

        last_time = time.time()
        correct = 0.

        if not params.linear_decay:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
                and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
            print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

        processed_training_samples = 0
        start_time = time.time()
        total_time = 0
        total_batch_generation_time = 0
        total_forward_time = 0
        total_backward_time = 0
        total_step_time = 0
        last_processed_training_samples = 0

        nonlocal processed_batches, stop_training, no_improvement, min_val_loss, losses, min_loss_criterion
        for i, (X_batch, tgt_batch) in enumerate(cbow_train_loader):

            batch_generation_time = (time.time() - start_time) * 1000000

            # forward pass
            forward_start = time.time()
            loss, k = forward_pass(X_batch, tgt_batch, params)
            all_costs.append(loss.item())
            forward_total = (time.time() - forward_start) * 1000000

            # backward
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()

            backward_total = (time.time() - backward_start) * 1000000

            # linear learning rate decay
            if params.linear_decay:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] - lr_shrinkage if \
                    'sgd' in params.optimizer else optimizer.param_groups[0]['lr']

            # optimizer step
            step_time = time.time()
            optimizer.step()
            total_step_time += (time.time() - step_time) * 1000000

            # log progress
            processed_training_samples += params.batch_size
            percentage_done = float(processed_training_samples) / num_training_samples
            processed_batches += 1
            if processed_batches == params.validation_frequency:
    
                # compute validation loss and train loss
                val_loss = round(validate(cbow_test_loader), 5) if num_val_samples > 0 else float('inf')
                train_loss = round(np.mean(all_costs), 5)

                # print current loss and processing speed
                logs.append('Epoch {3} - {4:.4} ; lr {2:.4} ; train-loss {0} ; val-loss {5} ; sentence/s {1}'.format(train_loss, int((processed_training_samples - last_processed_training_samples) / (time.time() - last_time)), optimizer.param_groups[0]['lr'], epoch, percentage_done, val_loss))
                if params.VERBOSE: 
                    print('\n\n\n')
                print(logs[-1])
                last_time = time.time()
                words_count = 0
                all_costs = []
                last_processed_training_samples = processed_training_samples
                
                if params.VERBOSE:
                    print("100 Batches took {} microseconds".format(total_time))
                    print("get_batch: {} \nforward: {} \nbackward: {} \nstep: {}".format(total_batch_generation_time / total_time, total_forward_time / total_time, total_backward_time / total_time, total_step_time / total_time))
                total_time = 0
                total_batch_generation_time = 0
                total_forward_time = 0
                total_backward_time = 0
                total_step_time = 0
                processed_batches = 0

                # save losses for logging later
                losses.append((train_loss, val_loss))

                # early stopping?
                if val_loss < min_val_loss:
                    min_val_loss = val_loss

                    # save best model
                    torch.save(cbow_net, os.path.join(params.outputdir, outputmodelname + '.cbow_net'))

                if params.stop_criterion is not None:
                    stop_crit_loss = eval(params.stop_criterion)
                    if stop_crit_loss < min_loss_criterion:
                        no_improvement = 0
                        min_loss_criterion = stop_crit_loss
                    else:
                        no_improvement += 1
                        if no_improvement > params.patience:
                            stop_training = True
                            print("No improvement in loss criterion", str(params.stop_criterion), 
                                  "for", str(no_improvement), "steps. Terminate training.")
                            break

            now = time.time() 
            batch_time_micro = (now - start_time) * 1000000

            total_time = total_time + batch_time_micro
            total_batch_generation_time += batch_generation_time
            total_forward_time += forward_total
            total_backward_time += backward_total
            
            start_time = now


    """
    Train model on CBOW objective
    """
    epoch = 1

    processed_batches = 0
    min_val_loss = float('inf')
    min_loss_criterion = float('inf')
    no_improvement = 0
    losses = []
    while not stop_training and epoch <= params.n_epochs:
        trainepoch(epoch)
        epoch += 1

    # load the best model
    if min_val_loss < float('inf'):
        cbow_net = torch.load(os.path.join(params.outputdir, outputmodelname + '.cbow_net'))
        print("Loading model with best validation loss.")
    else:
        # we use the current model;
        print("No model with better validation loss has been saved.")

    # save word vocabulary and counts
    pickle.dump(word_vec_copy, open( os.path.join(params.outputdir, outputmodelname + '.vocab'), "wb" ))

    if use_multiple_gpus:
        cbow_net = cbow_net.module
    return cbow_net.encoder, losses

def get_params_parser():

    parser = argparse.ArgumentParser(description='Training a word2mat model.')

    # paths
    parser.add_argument('--precomputed_word_vocab', type=str, default=None, help= \
        "Specify path where to load precomputed word.")
    parser.add_argument('--precomputed_chunks_dir', type=str, default=None, help= \
        "Specify path from where to load the chunkified input text.")
    parser.add_argument('--temp_path', type=str, required=True, help= \
        "Specify path where to save the chunkified input text.")

    # training parameters
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--linear_decay", action="store_true", help="If set, the learning rate is shrunk linearly after each batch as to approach minlr.")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument("--validation_frequency", type=int, default=500, help="How many batches to process before evaluating on the validation set (500).")
    parser.add_argument("--validation_fraction", type=float, default=0.0001, help="What fraction of the corpus to use for validation.\
                         Set to 0 to not use validation set based saving of intermediate models.")
    parser.add_argument("--stop_criterion", type=str, default=None, help="Which loss to use as stopping criterion.", choices = ['val_loss', 'train_loss'])
    parser.add_argument("--patience", type=int, default=3, help="How many validation steps to make before terminating training.")
    parser.add_argument("--VERBOSE", action="store_true", default=False, help="Whether to print additional info on speed of processing.")
    parser.add_argument("--num_workers", type=int, default=10, help="How many worker threads to use for creating the samples from the dataset.")

    # Word2Mat specific
    parser.add_argument("--w2m_type", type=str, default='cmow', choices=['cmow', 'cbow', 'hybrid'], help="Choose the encoder to use.")
    parser.add_argument("--word_emb_dim", type=int, default=100, help="Dimensionality of word embeddings.")
    parser.add_argument("--initialization", type=str, default='identity', help="Initialization strategy to use.", choices = ['one', 'identity', 'normalized', 'normal'])

    # dataset and vocab
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to a directory containing all files to use for training. " \
                        "One sentence per line in a file is assumed.")
    parser.add_argument("--max_words", type=int, default=None, help="Only produce embeddings for the most common tokens.")
    parser.add_argument("--num_docs", type=int, default=None, help="How many documents to consider from the source directory.")

    # CBOW specific
    parser.add_argument("--context_size", type=int, default=5, help="Context window size for CBOW.")
    parser.add_argument("--num_samples_per_item", type=int, default=1, help="Specify number of samples to generate from each sentence (the higher, the faster training).")
    parser.add_argument("--mode", type=str, help="Determines the mode of the prediction task, i.e., which word is to be removed from a given window of words. Options are 'cbow' (remove middle word) and 'random' (a random word from the window is removed).", default='random', choices = ['cbow', 'random'])
    parser.add_argument("--n_negs", type=int, default=5, help="How many negative samples to use for training (the larger the dataset, the fewer are required (5).")

    return parser

def prepare(params_senteval, samples):
    
    params = params_senteval["cmd_params"]
    outputmodelname = construct_model_name(params.outputmodelname, params)

    # Load vocabulary
    vocabulary = pickle.load(open(os.path.join(params.outputdir, outputmodelname + '.vocab'), "rb" ))[0]

    params_senteval['vocabulary'] = vocabulary
    params_senteval['inverse_vocab'] = {vocabulary[w] : w for w in vocabulary}

def _batcher_helper(params, batch):
    sent, _ = get_index_batch(batch, params.vocabulary)
    sent_cuda = Variable(sent.cuda())
    sent_cuda = sent_cuda.t()
    params.word2mat.eval() # Deactivate drop-out and such
    embeddings = params.word2mat.forward(sent_cuda).data.cpu().numpy()

    return embeddings

def batcher_cbow(params_senteval, batch):

    params = params_senteval["cmd_params"]
    embeddings = _batcher_helper(params_senteval, batch)    
    return embeddings

if __name__ == "__main__":

    run_and_evaluate(run_experiment, get_params_parser, batcher_cbow, prepare)
