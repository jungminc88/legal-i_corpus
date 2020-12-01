# -*- coding: utf-8 -*-
import os
import csv
import glob
import pickle
import pandas as pd
import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(2)
from transformers import BertJapaneseTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn import metrics
#####################################################################
# Helper functions to make the code more readable.


def argmax(vecs):
    # return the argmax as a python int
    _, idx = torch.max(vecs, 1)
    return idx

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vecs):
    max_score = torch.tensor([vecs[i,argmax(vecs)[i]] for i in range(len(vecs))],device=device)
    max_score_broadcast = max_score.view(BATCH_SIZE, -1).expand(BATCH_SIZE, vecs.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vecs - max_score_broadcast),1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
####################################################################
start_time = time.time()
####################################################################

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# prepare datasets


class Dataset(data.Dataset):
    def __init__(self, o_data):
        self.data = o_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index][0], self.data[index][1], self.data[index][2]
def collate_fn(batch):
    docs, masks, labels = zip(*batch)
    padded_docs, padded_masks, padded_labels = pad_sequence(docs), pad_sequence(masks), pad_sequence(labels)
    return padded_docs, padded_masks, padded_labels
"""
try:
    train_dataset, val_dataset, test_dataset, label_to_id = pickle.load(open("datasets.pkl", "rb"))
    print("dataset ready")
except (OSError, IOError) as e:
    directory_train = "/cl/work/jungmin-c/RSC_sentence_based_original/"
    directory_test = "/cl/work/jungmin-c/RSC_sentence_based_original/dev/"

    def get_dataset(directory):
        data = []
        label_to_id = {'BACKGROUND': 6, 'CONCLUSION': 1, 'FACT': 2, 'FRAMING-main': 3, 'FRAMING-sub': 4, 'IDENTIFYING': 5, 'OTHER': 0}
        START_LABEL = "<START>"
        STOP_LABEL = "<STOP>"
        label_to_id[START_LABEL]=len(label_to_id)
        label_to_id[STOP_LABEL]=len(label_to_id)
        files = glob.glob(os.path.join(directory, '[!(train)][!(dev)][!(alldata)][!(out)]*.csv'))
        for f in files:
            df = pd.read_csv(f, header=0, names=['sentence','label'])
            sentences = df.sentence.values
            labels = df.label.values
            labels = [label_to_id[label] for label in labels]
            
            filtered_input_ids = []
            attention_masks = []
            filtered_labels = []

            for i, sent in enumerate(sentences):
                if type(sent)==str:
                    encoded_dict = tokenizer.encode_plus(
                                        sent,                      
                                        add_special_tokens = True, # Special Tokenの追加
                                        max_length = 16,           # 文章の長さを固定（Padding/Trancatinating）
                                        pad_to_max_length = True,# PADDINGで埋める
                                        return_attention_mask = True,   # Attention maksの作成
                                        return_tensors = 'pt',     #  Pytorch tensorsで返す
                                        truncation = True
                                )
                    filtered_input_ids.append(encoded_dict['input_ids'])
                    filtered_labels.append(labels[i])
                    attention_masks.append(encoded_dict['attention_mask'])
            input_ids = torch.cat(filtered_input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            labels = torch.tensor(filtered_labels)
            data.append([input_ids, attention_masks, labels])
        dataset = Dataset(data)
        return dataset, label_to_id

    from torch.utils.data import TensorDataset, random_split
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    train_val_dataset, label_to_id = get_dataset(directory_train)
    # 90%地点のIDを取得
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size

    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    test_dataset, _ = get_dataset(directory_test)
    datasets = (train_dataset, val_dataset, test_dataset, label_to_id)
    pickle.dump( datasets, open( "datasets.pkl", "wb" ) )

    train_dataloader = DataLoader(train_dataset, batch_size=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, drop_last=True)

"""
BATCH_SIZE = 1
train_dataloader, test_dataloader = pickle.load(open("dataloaders.pkl", "rb"))
train_dataset, val_dataset, test_dataset, label_to_id = pickle.load(open("datasets.pkl", "rb"))


#####################################################################
# Create model

model_path = 'saved_model'
bert_model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking", # 日本語Pre trainedモデルの指定
    num_labels = 7, # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
    output_attentions = False, # アテンションベクトルを出力するか
    output_hidden_states = True # 隠れ層を出力するか
)
bert_model = nn.DataParallel(bert_model)
bert_model.cuda()
bert_model.load_state_dict(torch.load(model_path))

class BERT_CRF(nn.Module):

    def __init__(self, label_to_id, embedding_dim, hidden_dim):
        super(BERT_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_to_id = label_to_id
        self.labelset_size = len(label_to_id)   

        self.bert = bert_model
        self.e2f = nn.Linear(768, 9)
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.labelset_size, self.labelset_size)).to(device)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[label_to_id[START_LABEL], :] = -10000
        self.transitions.data[:, label_to_id[STOP_LABEL]] = -10000


    def _forward_alg(self, feats, device):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((BATCH_SIZE, self.labelset_size), -10000.)
        # START_TAG has all of the score.
        for i in range(BATCH_SIZE):
            init_alphas[i][self.label_to_id[START_LABEL]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.to(device)


        # Iterate through the doc
        for feat in feats: #feat[0] is the bunch of first sentences in the batch
            alphas_t = []  # The forward tensors at this timestep
            for next_label in range(self.labelset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_label].view(
                    BATCH_SIZE, -1).expand(BATCH_SIZE, self.labelset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_label].view(1,-1).expand(BATCH_SIZE,-1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_label_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_label_var).view(BATCH_SIZE))
            forward_var = torch.transpose(torch.stack(alphas_t),0,1).view(BATCH_SIZE, -1)
            del feat
            torch.cuda.empty_cache()
        terminal_var = forward_var + self.transitions[self.label_to_id[STOP_LABEL]].view(1,-1).expand(BATCH_SIZE,-1)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_bert_features(self, doc, mask):
        with torch.no_grad(): # 勾配計算なし
            all_encoder_layers = self.bert(doc, mask)
        embeddings_tok = all_encoder_layers[1][-2]
        embeddings = torch.mean(embeddings_tok, dim=1)

        feats = self.e2f(embeddings)
        feats = feats.unsqueeze(1)
        return feats
    def _score_doc(self, feats, labels):
        # Gives the score of a provided tag sequence
        score = torch.zeros(BATCH_SIZE,device=labels.device)
        #labels = torch.cat((torch.tensor([[self.label_to_id[START_LABEL]]],dtype=torch.long, device=labels.device).expand(1,BATCH_SIZE), labels),dim=0)
        labels = torch.cat([torch.tensor([self.label_to_id[START_LABEL]], dtype=torch.long, device=labels.device), labels])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[labels[i + 1].item(), labels[i].item()]+ feat[:,labels[i + 1].item()]
        score = score + self.transitions[self.label_to_id[STOP_LABEL], labels[-1]]
        return score

    def neg_log_likelihood(self, doc, mask, labels):
        feats = self._get_bert_features(doc, mask)
        #print(feats.shape)#[407,4,9]
        forward_score = self._forward_alg(feats,device)
        #print(forward_score)# [4]
        gold_score = self._score_doc(feats, labels)
        return forward_score - gold_score
        #[4]

    def _viterbi_decode(self, feats, device):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.labelset_size), -10000., device=device) 
        for i in range(BATCH_SIZE):
            init_vvars[i][self.label_to_id[START_LABEL]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_label in range(self.labelset_size):

                next_label_var = forward_var + self.transitions[next_label]
                best_label_id = argmax(next_label_var)
                bptrs_t.append(best_label_id)
                viterbivars_t.append(next_label_var[0][best_label_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.label_to_id[STOP_LABEL]]
        best_label_id = argmax(terminal_var)
        path_score = terminal_var[0][best_label_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_label_id]
        for bptrs_t in reversed(backpointers):
            best_label_id = bptrs_t[best_label_id]
            best_path.append(best_label_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.label_to_id[START_LABEL]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    def forward(self, doc, mask):  
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(doc, mask).to(device)
        #print("lstm_feats is "+ str(lstm_feats.size))#
        # Find the best path, given the features.
        score, label_seq = self._viterbi_decode(bert_feats,device)
        return score, label_seq

#####################################################################
# Run training


START_LABEL = "<START>"
STOP_LABEL = "<STOP>"
EMBEDDING_DIM = 50
HIDDEN_DIM = 4


model = BERT_CRF(label_to_id, EMBEDDING_DIM, HIDDEN_DIM)
model = model.to(device)

optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if not name.startswith('bert.bert')], lr=2e-5)


for epoch in range(1):
    #iterator = iter(train_dataloader)
    #for step, (docs, tagsets) in enumerate(iterator):
    for doc, mask, labels in train_dataset:

        doc = doc.to(device)
        mask = mask.to(device)
        labels = labels.to(device)


        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before eaquich instance
        model.zero_grad()

        # Step 2. Run our forward pass.
        loss = model.neg_log_likelihood(doc, mask, labels)
        del doc, mask, labels
        torch.cuda.empty_cache()
        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.mean().backward()
        print(loss)
        optimizer.step()

print("---%s seconds ---" % (time.time() - start_time))

def test_loop(test_dataset, model):
    Pred = list()
    Gold = list()
    model.eval()   # set to evaluation mode
    for batch in test_dataset:
        # Compute the loss without gradient calculation
        with torch.no_grad():
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            _, preds = model(b_input_ids, b_input_mask)
            Pred.extend(preds)
            Gold.extend(b_labels)

        # Make sure to free GPU memory
        del preds
        torch.cuda.empty_cache()
    Pred = torch.stack(Pred).squeeze(1)
    Pred = Pred.cpu().data.numpy()
    Gold = torch.stack(Gold)
    Gold = Gold.cpu().data.numpy()

    print(metrics.precision_recall_fscore_support(Gold, Pred, average='macro'))
    print(metrics.precision_recall_fscore_support(Gold, Pred, average='micro'))
    print(metrics.precision_recall_fscore_support(Gold, Pred, average='weighted'))
    accuracy = metrics.accuracy_score(Gold, Pred)
    print("Test:\tAccuracy:{}".format(accuracy))

test_loop(test_dataset, model)
