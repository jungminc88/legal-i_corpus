# -*- coding: utf-8 -*-

import os
import csv

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

#####################################################################
# Helper functions to make the code more readable.


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
####################################################################
# prepare datasets
import os
import csv
import torch
import numpy as np
from gensim.models.word2vec import Word2Vec

model_path = '/home/cl/jungmin-c/japanese-word2vec-model-builder/output/word2vec.gensim.model'
lang_model = Word2Vec.load(model_path)
directory_train = "/home/cl/jungmin-c/legal-i_corpus/AIL2019/RSC_original2/"
directory_test = "/home/cl/jungmin-c/legal-i_corpus/test_hanrei/"
tag_to_ix = {}
def build_emb(directory):
    data = []
    for item in os.listdir(directory):

        datafile = open(directory + item,'r')
        reader = csv.reader(datafile)
        next(reader)
        sentences = []
        tags = []
        for row in reader:
            sentence = row[0].split()
            sentence = list(filter(lambda x: x in lang_model.wv.vocab, sentence))
            if len(sentence):
                sent_emb = np.mean([lang_model[word] for word in sentence],axis=0)
                sentences.append(sent_emb)
                tag = row[1]
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)
                tags.append(tag_to_ix[tag])


        sentences = torch.tensor(sentences)
        tags = torch.tensor(tags)
        data.append((sentences, tags))

    return data
test_data = build_emb(directory_test)
train_data = build_emb(directory_train)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix[START_TAG]=len(tag_to_ix)
tag_to_ix[STOP_TAG]=len(tag_to_ix)

#####################################################################
# Create model

class BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)


        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, doc):
        self.hidden = self.init_hidden()
        self.hidden = (self.hidden[0].type(torch.FloatTensor).cuda(), self.hidden[1].type(torch.FloatTensor).cuda()) 

        embeds = doc.view(len(doc), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(doc), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_doc(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):

                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, doc, tags):
        feats = self._get_lstm_features(doc)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_doc(feats, tags)
        return forward_score - gold_score

    def forward(self, doc):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(doc).to(device)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#####################################################################
# Run training


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50
HIDDEN_DIM = 4


model = BiLSTM_CRF(tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(100):
    for doc, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        if(torch.cuda.is_available()):
            doc=doc.to(device)
            tags=tags.to(device)
        model.zero_grad()

        # Step 2. Run our forward pass.
        loss = model.neg_log_likelihood(doc, tags)

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
check = test_data[0][0]
check = check.to(device)
print(model(check))