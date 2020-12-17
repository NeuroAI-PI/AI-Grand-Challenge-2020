# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ######################################################################
# 형태소분석 기반 BERT 모델 문서분류 Fine-tuning 샘플
# (original: Hugging-face BERT example code)
# 수정: kyoungman.bae
# 일자: 2019-05-30
#

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassificationMLP1, \
    BertForSequenceClassificationMLP2, BertForSequenceClassificationMLP3,\
    BertForSequenceClassificationCNN, BertForSequenceClassificationLSTM, BertConfig, \
    WEIGHTS_NAME, CONFIG_NAME
### kyoungman.bae @ 19-05-28
from pytorch_pretrained_bert.tokenization_morp import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

import urllib3
import json
from sklearn.metrics import f1_score, accuracy_score
from collections import OrderedDict

result_json = OrderedDict()

# ------------------------------------------------------------------------------------------#
import pandas as pd
import pickle
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sentence_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='UTF8') as f:
            lines = pd.read_csv(f, sep='\t')

            return lines

    @classmethod
    def _read_excel(cls, input_file, sheet_name, quotechar=None):
        """Reads a tab separated value file."""
        lines = pd.read_excel(input_file, sheet_name=str(sheet_name))
        for i in range(len(lines)):
            lines['Script'][i] = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ·!』\\‘’|\(\)\[\]\<\>`\'…》]', '', lines['Script'][i])
        print(lines['Script'])
        return lines

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='UTF8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(np.unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class AI_Challenge_Processor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir, sheet_num, guid_s_num):
        """See base class."""
        return self._create_examples(
            self._read_excel(data_dir, sheet_num), sheet_num, guid_s_num,
            "train")

    def get_dev_examples(self, data_dir, sheet_num, guid_s_num):
        """See base class."""
        return self._create_examples(
            self._read_excel(os.path.join(data_dir, "test_text.xlsx"), sheet_num), sheet_num, guid_s_num, "dev")

    def get_labels(self, data_dir):
        """See base class."""
        ######################################################################
        ### kyoungman.bae @ 19-05-30 @ for multi label classification
        ### You need to add a file with label information in the data folder.
        ### You should use a numbered label on a line.
        labels = []
        lines = self._read_tsv(os.path.join(data_dir, "labels.tsv"))
        for (i, line) in enumerate(lines):
            labels.append(str(line[0]))
        return labels

    def _create_examples(self, lines, sheet_num, guid_s_num, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        full_text = lines['Script']
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, guid_s_num + i)
            text_a = full_text[i]
            label = str(sheet_num)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_tokens(examples, label_list, tokenizer, openapi_key_dir):

    label_map = {label: i for i, label in enumerate(label_list)}
    result = []

    f = open(openapi_key_dir, 'r')
    openapi_key_list = f.read().split()
    key_num = 0
    openapi_key = openapi_key_list[key_num]
    for (ex_index, example) in enumerate(examples):
        tokens_a = do_lang(openapi_key, example.text_a)

        while "openapi error" in tokens_a:
            key_num += 1
            openapi_key = openapi_key_list[key_num]
            tokens_a = do_lang(openapi_key, example.text_a)

        tokens_a = tokenizer.tokenize(tokens_a)
        label_id = label_map[example.label]
        result.append({'tokens': tokens_a, 'label': label_id})

    return result


def convert_tokens_to_features(tokens_list, max_seq_length, tokenizer):

    features = []

    for tokens_idx in range(len(tokens_list)):

        tokens_a = tokens_list[tokens_idx]['tokens']
        label_id = tokens_list[tokens_idx]['label']

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def convert_tokens_to_features_eval(tokens_list, max_seq_length, tokenizer, overlap_size_token):

    features, len_tokens_list, label_list, num_tokens_list = [], [], [], []

    for tokens_idx in range(len(tokens_list)):

        tokens_a = tokens_list[tokens_idx]['tokens']
        label_id = tokens_list[tokens_idx]['label']
        label_list.append(label_id)

        temp_tokens = tokens_a
        loop = 0
        max_num_token = max_seq_length-2
        final_tokens_list = []
        len_tokens_list.append(len(temp_tokens))

        while len(temp_tokens) > max_num_token:
            loop += 1
            temp_tokens = temp_tokens[:max_num_token]
            tokens = ["[CLS]"] + temp_tokens + ["[SEP]"]
            final_tokens_list.append(tokens)
            temp_tokens = tokens_a[(max_num_token - overlap_size_token) * loop:]

        tokens = ["[CLS]"] + temp_tokens + ["[SEP]"]
        final_tokens_list.append(tokens)


        num_tokens_list.append(len(final_tokens_list))

        for token_num in range(len(final_tokens_list)):

            tokens = final_tokens_list[token_num]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

    return features, num_tokens_list, len_tokens_list, label_list

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


##########################################################
### kyoungman.bae @ 2019-05-30
### do morphology analysis using OpenAPI service
def do_lang(openapi_key, text):
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"

    requestJson = {"access_key": openapi_key, "argument": {"text": text, "analysis_code": "morp"}}

    http = urllib3.PoolManager()
    response = http.request("POST", openApiURL, headers={"Content-Type": "application/json; charset=UTF-8"},
                            body=json.dumps(requestJson))

    json_data = json.loads(response.data.decode('utf-8'))
    json_result = json_data["result"]

    if json_result == -1:
        json_reason = json_data["reason"]
        if "Invalid Access Key" in json_reason:
            logger.info(json_reason)
            logger.info("Please check the openapi access key.")
            sys.exit()
        return "openapi error - " + json_reason
    else:
        json_data = json.loads(response.data.decode('utf-8'))

        json_return_obj = json_data["return_object"]

        return_result = ""
        json_sentence = json_return_obj["sentence"]
        for json_morp in json_sentence:
            for morp in json_morp["morp"]:
                return_result = return_result + str(morp["lemma"]) + "/" + str(morp["type"]) + " "

        return return_result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--openapi_key_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--classification_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    ### kyoungman.bae @ 19-05-28 @
    parser.add_argument("--bert_model_path", default=None, type=str, required=True,
                        help="Bert pre-trained model path")
    parser.add_argument("--sheet_num",
                        default=5,
                        type=int,
                        help="violence excel sheet number")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        # default=256,
                        default=128,
                        # default=62,
                        # default=40,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--overlap_size_token",
                        default=64,
                        # default=30,
                        type=int)
    parser.add_argument("--non_violence_threshold",
                        # default=0.6,
                        default=0.1,
                        type=int)
    parser.add_argument("--test_code",
                        default=2,
                        type=int,
                        help="0 : Sample Text / 1 : Google STT / 2 : Google STT (+non_violence)")
    parser.add_argument("--voting_code",
                        default=1,
                        type=int,
                        help="0 : threshold / 1 : range / 2 : all")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--new_train_data",
                        action='store_true',
                        help="Whether to use new train data.")
    parser.add_argument("--do_plus_sample",
                        action='store_true',
                        help="Whether to plus sample text data.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        # default=6,
                        # default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        # default=8,
                        default=6,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args(['--data_dir',
                              './dataset/',
                              '--eval_model_dir',
                              './eval_model/',
                              '--openapi_key_dir',
                              './requirements/openapi_key_list.txt',
                              '--task_name', 'AI_Challenge',
                              '--classification_name', 'cnn',
                              '--output_dir',
                              './output/',
                              '--vocab_file',
                              './requirements/vocab.korean_morp.list',
                              '--bert_model_path',
                              './requirements/',
                              '--do_eval', '--do_train'])
    # '--do_eval' / '--do_train' / '--new_train_data' / '--do_plus_sample

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    classification_name = args.classification_name

    if classification_name == "mlp1":
        BertForSequenceClassification = BertForSequenceClassificationMLP1
    elif classification_name == "mlp2":
        BertForSequenceClassification = BertForSequenceClassificationMLP2
    elif classification_name == "mlp3":
        BertForSequenceClassification = BertForSequenceClassificationMLP3
    elif classification_name == "cnn":
        BertForSequenceClassification = BertForSequenceClassificationCNN
    elif classification_name == "lstm":
        BertForSequenceClassification = BertForSequenceClassificationLSTM

    processors = {
        "ai_challenge": AI_Challenge_Processor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    ### kyoungman.bae @ 19-05-28

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    # load train tokens
    with open('./features/train_tokens.pickle', "rb") as fr:
        train_tokens = pickle.load(fr)

    # using ETRI API for train new data
    if args.new_train_data:
        train_examples = []
        if args.do_train:
            for s_num in range(args.sheet_num):
                train_examples.extend(processor.get_train_examples(args.data_dir + "new_data/file_name",
                                                                   s_num, len(train_examples)))

            new_train_tokens = convert_examples_to_tokens(train_examples, label_list, tokenizer, args.openapi_key_dir)
            train_tokens.extend(new_train_tokens)

            # save
            with open('./features/file_name', 'wb') as fw:
                pickle.dump(train_tokens, fw, pickle.HIGHEST_PROTOCOL)

    if args.do_plus_sample:
        # load sample text tokens
        with open('./features/file_name', "rb") as fr:
            sample_tokens = pickle.load(fr)

        train_tokens.extend(sample_tokens)

    num_train_optimization_steps = int(
        len(train_tokens) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model_path,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step, nb_tr_steps, tr_loss = 0, 0, 0
    if args.do_train:
        train_features = convert_tokens_to_features(train_tokens, args.max_seq_length, tokenizer)

        if len(train_features) == 0:
            logger.info("The number of train_features is zero. Please check the tokenization. ")
            sys.exit()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))

    else:
        eval_model_file = os.path.join(args.eval_model_dir, WEIGHTS_NAME)
        eval_config_file = os.path.join(args.eval_model_dir, CONFIG_NAME)
        config = BertConfig(eval_config_file)
        model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(eval_model_file))
    model.to(device)

    eval_examples = []
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # 0: Sample Text / 1 : Google STT / 2 : Google STT (+non_violence)
        # Sample Text / test_code : 0
        if args.test_code == 0:
            with open("./features/file_name", "rb") as fr:
                eval_examples = pickle.load(fr)

        # Google STT / test_code : 1
        elif args.test_code == 1:
            with open("./features/file_name", "rb") as fr:
                eval_examples = pickle.load(fr)

        # Google STT (+non_violence) / test_code : 2
        elif args.test_code == 2:
            with open("./features/file_name", "rb") as fr:
                eval_examples = pickle.load(fr)

        eval_features, num_features, len_features, eval_label = convert_tokens_to_features_eval(eval_examples,
                                                                                           args.max_seq_length,
                                                                                           tokenizer,
                                                                                           args.overlap_size_token)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # print(all_input_ids.shape)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        outputs, all_pred_out = [], []

        temp_num = 0
        output_label_list, final_pred_out = [], []
        class_code = ["000001", "020121", "02051", "020811", "020819"]

        ##################################################
        ### kyoungman.bae @ 19-05-31
        ### The classified result labels are displayed in the following path.
        output_eval_file = os.path.join(args.output_dir, "file_name")
        with open(output_eval_file, "w") as writer:
            for num in range(len(num_features)):
                input_ids = all_input_ids[temp_num: temp_num + num_features[num], ].to(device)
                input_mask = all_input_mask[temp_num: temp_num + num_features[num], ].to(device)
                segment_ids = all_segment_ids[temp_num: temp_num + num_features[num], ].to(device)
                label_ids = all_label_ids[temp_num: temp_num + num_features[num], ].to(device)
                temp_num = temp_num + num_features[num]

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                ### kyoungman.bae @ 19-05-3
                current_out = np.argmax(logits, axis=1)
                all_pred_out.extend(current_out)

                # tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                # eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

                mean_logits = logits.sum(axis=0) / len(logits)
                max_idx = np.argmax(mean_logits)

                # start test voting
                # test_threshold / voting_code : 0
                if args.voting_code == 0:

                    if mean_logits[0] > args.non_violence_threshold:
                        output_label = np.argmax(mean_logits)
                    else:
                        output_label = np.argmax(mean_logits[1:]) + 1

                # test_range / voting_code : 1
                elif args.voting_code == 1:

                    if max_idx == 0:
                        second_idx = np.argmax(mean_logits[1:])

                        if mean_logits[max_idx] - mean_logits[second_idx] < args.non_violence_threshold:
                            output_label = second_idx + 1
                        else:
                            output_label = max_idx
                    else:
                        output_label = max_idx

                # test_all / voting_code : 2
                elif args.voting_code == 2:

                    for i in range(len(logits)):
                        if np.argmax(logits[i]) == 0:
                            non_violence_flag = True
                            continue
                        else:
                            non_violence_flag = False
                            logits = logits[:, 1:]
                            break
                    if non_violence_flag:
                        output_label = 0
                    else:
                        mean_logits = logits.sum(axis=0) / len(logits)
                        output_label = np.argmax(mean_logits) + 1

                final_pred_out.append(output_label)
                writer.write("%s\n" % (int(output_label)))
                file_name = []
                wav_num = str(num + 1)

                if len(wav_num) == 1:
                    file_name = "t2_000" + wav_num + ".wav"
                elif len(wav_num) == 2:
                    file_name = "t2_00" + wav_num + ".wav"
                elif len(wav_num) == 3:
                    file_name = "t2_0" + wav_num + ".wav"

                output_label_list.append({
                    "id": num,
                    "file_name": file_name,
                    "class code": class_code[output_label]
                })
            result_json["annotations"] = output_label_list

        eval_loss = eval_loss / nb_eval_steps
        all_label_ids = all_label_ids.tolist()
        eval_f1_score = f1_score(eval_label, final_pred_out, average='macro')
        eval_accuracy = accuracy_score(eval_label, final_pred_out)
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'eval_f1_score': eval_f1_score}

        output_eval_file = os.path.join(args.output_dir, "file_name")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        with open('result.json', 'w', encoding="utf-8") as make_file:
            json.dump(result_json, make_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
