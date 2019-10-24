"""
Train a model on TACRED.
"""

import os
import sys
from datetime import datetime
import time
import numpy as np
import random 
import argparse
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim 

from data.loader import DataLoader
from model.trainer import LSTMTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

def get_parser():
    parser = argparse.ArgumentParser()
    # 待修改，只适用于TACRED
    parser.add_argument("--data_dir", type=str, default='dataset/tacred', help="Folder of tacred dataset")
    parser.add_argument("--vocab_dir", type=str, default="dataset/vocab", help="Folder of vocabulary data")

    # Embedding hyper-parameter
    parser.add_argument("--emb_dim", type=int, default=300, help="Word embedding dimension.")
    parser.add_argument("--ner_dim", type=int, default=30, help="NER embedding dimension.")
    parser.add_argument("--pos_dim", type=int, default=30, help="POS embedding dimension.")

    # model parameter
    parser.add_argument("--hidden_dim", type=int, default=200, help="RNN hidden state size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of RNN layers.")
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
    parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
    parser.add_argument('--no-attn', dest='attn', action='store_false')
    parser.set_defaults(attn=True)
    parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
    parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')
    # is lower?
    parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
    parser.add_argument('--no-lower', dest='lower', action='store_false')
    parser.set_defaults(lower=False)


    parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
    parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

    args = parser.parse_args()
    return args

def main():
    args = get_parser()

    # set seed and prepare for training 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(1234)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    init_time = time.time()

    # make opt
    opt = vars(args)
    label2id = constant.LABEL_TO_ID
    opt['num_class'] = len(label2id)

    # load vocab and word embedding for training
    vocab_file = os.path.join(opt['vocab_dir'], 'vocab.pkl')
    vocab = Vocab(vocab_file, load=True)
    opt['vocab_size'] = vocab.size
    emb_file = os.path.join(opt['vocab_dir'], 'embedding.npy')
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == vocab.size
    assert emb_matrix.shape[1] == opt['emb_dim']

    # load data
    print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
    train_batch = DataLoader(os.path.join(opt['data_dir'], 'train.json'), opt['batch_size'], opt, vocab, evaluation=False)
    dev_batch = DataLoader(os.path.join(opt['data_dir'], 'dev.json'), opt['batch_size'], opt, vocab, evaluation=True)

    model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + str(model_id)
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
  
    # save config
    path = os.path.join(model_save_dir, 'config.json')
    helper.save_config(opt, path, verbose=True)
    vocab.save(os.path.join(model_save_dir, 'vocab.pkl'))
    file_logger = helper.FileLogger(os.path.join(model_save_dir, opt['log']), 
                                    header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

    
    # print model info
    helper.print_config(opt)

    # Build Model
    if not opt['load']:
        trainer = LSTMTrainer(opt, emb_matrix)
    else:
        model_file = opt['model_file']
        print("Loading model from {}".format(model_file))
        model_opt = torch_utils.load_config(model_file)
        model_opt['optim'] = opt['optim']
        trainer = LSTMTrainer(model_opt)
        trainer.load(model_file)

    id2label = dict([(v, k) for k, v in label2id.items()])
    dev_score_history = []
    current_lr = opt['lr']

    global_step = 0
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_batch) * opt['num_epoch']

    # start training
    for epoch in range(1, opt['num_epoch'] + 1):
        train_loss = 0
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch)
            train_loss += loss
            if global_step % opt['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                    opt['num_epoch'], loss, duration, current_lr))
        
        # eval on dev
        print("Evaluating on dev set ...")
        predictions = []
        dev_loss = 0.0
        for i, batch in enumerate(dev_batch):
            preds, _, loss = trainer.predict(batch)
            predictions += preds
            dev_loss += loss 
        predictions = [id2label[p] for p in predictions]
        train_loss = train_loss / train_batch.num_examples * opt['batch_size']
        dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

        dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
        print("epoch {}: train loss = {:.6f}, dev loss = {:.6f}, dev f1 = {:.4f}".format(
            epoch, train_loss, dev_loss, dev_f1
        ))
        dev_score = dev_f1
        file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(
            epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))
        
        # save model
        model_file = os.path.join(model_save_dir, "checkpoint_epoch_{}.py".format(epoch))
        trainer.save(model_file, epoch)
        if epoch == 1 or dev_score > max(dev_score_history):
            copyfile(model_file, model_save_dir + '/best_model.pt')
            print("new best model saved.")
            file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
                .format(epoch, dev_p*100, dev_r*100, dev_score*100))
        if epoch % opt['save_epoch'] != 0:
            os.remove(model_file)
        
        if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
            current_lr *= opt['lr_decay']
            trainer.update_lr(current_lr)
        
        dev_score_history += [dev_score]
        print("")
    
    print("Training ended with {} epochs.".format(epoch))

if __name__ == "__main__":
    main()

    
    





