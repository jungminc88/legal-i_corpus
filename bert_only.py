import os
import torch
import torch.nn as nn
import pandas as pd
import pickle
from sklearn import metrics
from logging import getLogger
# GPUが使えれば利用する設定
print("starting")
import time
model_path = 'pytorch_model'
start_time = time.time()
logger = getLogger()
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is: {}".format(device))
def get_dataset(path):
    print("setting up dataloaders")
    df = pd.read_csv(path, header=0, names=['sentence','label']) 
    sentences = df.sentence.values
    labels = df.label.values
    dict_labels = {}
    for label in labels:
        if label not in dict_labels:
            dict_labels[label] = len(dict_labels)
    labels = [dict_labels[label] for label in labels]
    from transformers import BertJapaneseTokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    
    filtered_input_ids = []
    attention_masks = []
    filtered_labels = []
    for i, sent in enumerate(sentences):
        if type(sent)==str:
            encoded_dict = tokenizer.encode_plus(
                                sent,                      
                                add_special_tokens = True, # Special Tokenの追加
                                max_length = 512,           # 文章の長さを固定（Padding/Trancatinating）
                                pad_to_max_length = True,# PADDINGで埋める
                                return_attention_mask = True,   # Attention maksの作成
                                return_tensors = 'pt',     #  Pytorch tensorsで返す
                                truncation = True
                        )
            filtered_input_ids.append(encoded_dict['input_ids'])
            filtered_labels.append(labels[i])
            attention_masks.append(encoded_dict['attention_mask'])
    # リストに入ったtensorを縦方向（dim=0）へ結合
    input_ids = torch.cat(filtered_input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # tenosor型に変換
    labels = torch.tensor(filtered_labels)

    from torch.utils.data import TensorDataset, random_split
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    # データセットクラスの作成
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

try:
    train_dataloader, test_dataloader = pickle.load(open("dataloaders.pkl", "rb"))
    print("skipping dataloader setup")
except (OSError, IOError) as e:
    from torch.utils.data import TensorDataset, random_split
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    train_dataset = get_dataset("/cl/work/jungmin-c/RSC_sentence_based_original/train.csv")
    test_dataset = get_dataset("/cl/work/jungmin-c/RSC_sentence_based_original/dev/dev.csv")

    # データローダーの作成
    batch_size = 4


    # 訓練データローダー
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), # ランダムにデータを取得してバッチ化
                batch_size = batch_size,
                drop_last = True
            )


    #　テストデータローダー
    test_dataloader = DataLoader(
                test_dataset, 
                sampler = SequentialSampler(test_dataset), # 順番にデータを取得してバッチ化
                batch_size = 1
            )
            
    # pickle
    dataloaders = (train_dataloader, test_dataloader)
    pickle.dump( dataloaders, open( "dataloaders.pkl", "wb" ) )


from transformers import BertForSequenceClassification, BertConfig

# BertForSequenceClassification 学習済みモデルのロード

model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking", # 日本語Pre trainedモデルの指定
    num_labels = 7, # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
    output_attentions = False, # アテンションベクトルを出力するか
    output_hidden_states = False, # 隠れ層を出力するか
)
model = nn.DataParallel(model)
# モデルをGPUへ転送
model.cuda()

try:
    model.load_state_dict(torch.load(model_path),strict=False)
    print("load model data from " + str(model_path))
except Exception:
    pass

# 最適化手法の設定
optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if not name.startswith('bert.bert')], lr=2e-5)
#optimizer = optim.SGD([param for name, param in model.named_parameters() if not name.startswith('bert.bert')], lr=0.01, weight_decay=1e-4)


# 訓練パートの定義
def train(model):
    model.train() # 訓練モードで実行
    train_loss = 0
    for batch in train_dataloader:# train_dataloaderはword_id, mask, labelを出力する点に注意
        b_input_ids = batch[0].to(device) #[4,512]
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)#[4]
        optimizer.zero_grad()
        loss, logits = model(input_ids = b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    return train_loss


train_loss_ = []
for epoch in range(1):
    print("Epoch: {}".format(epoch))
    train_loss = train(model)
    train_loss_.append(train_loss)
print(train_loss_)


torch.save(model.state_dict(), model_path)


model.eval()
Pred = list()
Gold = list()
test_iter = iter(test_dataloader)
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():   
        # 学習済みモデルによる予測結果をpredsで取得     
        pred = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        Pred.extend(pred)
        Gold.extend(b_labels)

Pred = torch.stack(Pred).squeeze(1)
Pred = torch.argmax(Pred, dim=1)
Pred = Pred.cpu().data.numpy()
Gold = torch.stack(Gold)
Gold = Gold.cpu().data.numpy()

print(metrics.precision_recall_fscore_support(Gold, Pred, average='macro'))
print(metrics.precision_recall_fscore_support(Gold, Pred, average='micro'))
print(metrics.precision_recall_fscore_support(Gold, Pred, average='weighted'))

print("--- %s seconds ---" % (time.time() - start_time))
