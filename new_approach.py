import math
import torch
from torch import nn
#!pip install d2l
from d2l import torch as d2l
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt

############################
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = valid_lens.repeat(shape[1])

    X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-math.inf)

    return nn.functional.softmax(X.reshape(shape), dim=-1)

############################
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d=keys.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights=masked_softmax(X=scores,valid_lens=valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

############################
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        b,n,h=X.shape
        num_hiddens_per_head = h // self.num_heads
        X=X.reshape(b,n,self.num_heads,num_hiddens_per_head)
        return X

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        b,n,heads,num_hiddens_per_head=X.shape
        X=X.reshape(b,n,heads * num_hiddens_per_head)
        return X

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)

        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        #print(queries.shape,values.shape,keys.shape)

        queries=self.transpose_output(queries.permute(3,1,2,0)).permute(2,1,0)
        keys=self.transpose_output(keys.permute(3,1,2,0)).permute(2,1,0)
        values=self.transpose_output(values.permute(3,1,2,0)).permute(2,1,0)
        #print(queries.shape,values.shape,keys.shape)

        valid_lens=valid_lens.repeat(self.num_heads)

        output_concat=self.attention(queries,keys,values,valid_lens)
        output_concat=self.transpose_output(self.transpose_qkv(output_concat.permute(2,1,0)).permute(3,1,2,0))

        return self.W_o(output_concat)

############################
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P=torch.zeros(max_len,num_hiddens)
        for i in range(self.P.shape[0]):
          for j in range(0,self.P.shape[1]//2):
            self.P[i,2*j]=math.sin(i/10000**(2*j/num_hiddens))
            self.P[i,(2*j)+1]=math.cos(i/10000**(2*j/num_hiddens))
        self.P=self.P.unsqueeze(0)

    def forward(self, X):
        Y=self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X+Y)

############################
class FeedForward(nn.Module):
    """Feed Forward Network for encoder blocks"""
    def __init__(self, attn_hidden, ffn_num_hidden):
        super().__init__()
        self.linear1 = nn.Linear(attn_hidden, ffn_num_hidden)
        self.linear2 = nn.Linear(ffn_num_hidden, attn_hidden)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

#############################
class Encoder(nn.Module):
    """Encoder with multi-head attention,feed-forward network, addition and normalization."""
    def __init__(self, num_hiddens, num_heads, ffn_num_hiddens, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.feed_forward = FeedForward(num_hiddens, ffn_num_hiddens)
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.norm2 = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, valid_lens):
        from_attention_output = self.attention(X, X, X, valid_lens)
        X = self.norm1(X + self.dropout(from_attention_output))
        feed_forward_output = self.feed_forward(X)
        X = self.norm2(X + self.dropout(feed_forward_output))
        return X

#############################
class EncoderBlockModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_heads, ffn_num_hiddens, num_layers, dropout, num_classes,with_pos_encodings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        self.encoder = nn.Sequential(
            *[Encoder(num_hiddens, num_heads, ffn_num_hiddens, dropout) for n in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_hiddens, num_classes)
        self.with_pos_encodings = with_pos_encodings

    def forward(self, X, valid_lens):
        X = self.embedding(X)
        if self.with_pos_encodings:
            X += self.pos_encoding(X)
        for layer in self.encoder:
            X = layer(X, valid_lens)
        X = X.mean(dim=1)
        return self.classifier(self.dropout(X))


#######################################
###### Model training and evaluation###
#######################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Skip this if you have already downloaded the dataset
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz',
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')


#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data = []
    labels = []
    folder = 'train' if is_train else 'test'
    for label in ['pos', 'neg']:
        labeled_folder = os.path.join(data_dir, folder, label)
        for filename in os.listdir(labeled_folder):
            if filename.endswith('.txt'):
                with open(os.path.join(labeled_folder, filename), 'r', encoding='utf-8') as f:
                    review = f.read()
                    data.append(review)
                    labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = '../data/aclImdb'
    train_data = read_imdb(data_dir, is_train=True)
    test_data = read_imdb(data_dir, is_train=False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = DataLoader(TensorDataset(train_features, torch.tensor(train_data[1])), batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(TensorDataset(test_features, torch.tensor(test_data[1])), batch_size=batch_size)
    return train_iter, test_iter, vocab

def load_data_imdb2(batch_size, train_vocab, num_steps=500):
        df_test=pd.read_csv("test_data_movie.csv")
        test_data = (df_test['text'].astype(str).tolist(), df_test['label'].tolist())
        test_tokens = d2l.tokenize(test_data[0], token='word')
        test_features = torch.tensor([d2l.truncate_pad(
            vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
        test_iter = DataLoader(TensorDataset(test_features, torch.tensor(test_data[1])), batch_size=batch_size)
        return test_iter

def predict_sentiment2(net, vocab, sequence):
        """Predict the sentiment of a text sequence."""
        indices = [vocab[i] for i in sequence.split()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            temp = 0
        for i in range(len(indices)):
            if indices[i] != 0:
                temp+=1
        valid_lens = torch.Tensor([temp])
        prediction=net(torch.tensor(indices).unsqueeze(0).to(device),valid_lens.to(device))
        label = torch.argmax(prediction, dim=1).item()
        return 'positive' if label == 1 else 'negative'


def cal_metrics2(net, test_iter, vocab):
        """Outputs a CSV file 'prediction_results2.csv' with columns: review,gold_label,predicted_label.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        net.eval()

        results = []
        tp, tn, fp, fn = 0, 0, 0, 0
        pad_idx = vocab['<pad>']

        for batch_idx, (data, target) in enumerate(test_iter):
            valid_lens = (data != pad_idx).sum(dim=1).to(device)
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                prediction = net(data, valid_lens)
            label = torch.argmax(prediction, dim=1)
            data_cpu = data.cpu().numpy()
            for i in range(len(data_cpu)):
                tokens = [vocab.idx_to_token[idx] for idx in data_cpu[i] if idx != pad_idx]
                review_text = " ".join(tokens)
                gold_label = int(target[i].item())
                pred_label = int(label[i].item())
                results.append((review_text, gold_label, pred_label))

            tp += ((label == 1) & (target == 1)).sum().item()
            tn += ((label == 0) & (target == 0)).sum().item()
            fp += ((label == 1) & (target == 0)).sum().item()
            fn += ((label == 0) & (target == 1)).sum().item()

        accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0
        precision = tp/(tp+fp) if (tp+fp) > 0 else 0
        recall = tp/(tp+fn) if (tp+fn) > 0 else 0
        F1_Score = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0

        
        with open('prediction_results2.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["review", "gold_label", "predicted_label"])
            for review_text, gold_label, pred_label in results:
                writer.writerow([review_text, gold_label, pred_label])

        return F1_Score, precision, recall, accuracy

######################################################
train_or_infer=input("train or infer") #train and infer on new weights, or infer from pretrained weights
if train_or_infer=="train": #train and infer from trained weights
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb(batch_size)
    print(len(vocab))

    devices = d2l.try_all_gpus()
    net = EncoderBlockModel(vocab_size = len(vocab), embed_size = 100, num_hiddens = 100, num_heads = 5, ffn_num_hiddens = 256, num_layers = 1, dropout = 0.5, num_classes = 2) #add argument with_pos_encodings=True if you want positional encodings

    def init_weights(module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight)
        if type(module) == nn.LSTM:
            for param in module._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(module._parameters[param])
    net.apply(init_weights)

    #embeddings, num_hiddens = 100,100
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)

    lr, num_epochs = 0.001, 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=1e-4)
    net.to(device)
    loss_array=[]
    net.train()
    for n in range(num_epochs):
            train_loss = 0
            target_num = 0
            net.train()
            for batch_idx, (data, target) in enumerate(train_iter):
                data, target = data.to(device), target.to(device)
                loss = None
                valid_lens = (data != 0).sum(dim=1).to(device)
                prediction= net(data,valid_lens)
                loss = criterion(prediction,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * target.size(0)
                target_num += target.size(0)

            train_loss /= target_num
            loss_array.append(train_loss)
            print('Epoch: {}, Training Loss: {:.4f}'.format(n+1, train_loss))

    plt.plot([i+1 for i in range(num_epochs)],loss_array)
    plt.xlabel("Epoch")
    plt.ylabel("Train_loss")
    
    ###### Test on IMDB Test Dataset ######
    print("Test on IMDB Test Dataset")
    print("F1,Precision,Recall,Accuracy")
    print(cal_metrics2(net, test_iter, vocab))


    ###### Test on CSV File ########
    test_iter_final=load_data_imdb2(batch_size=64, train_vocab=vocab, num_steps=500)
    print("Test on CSV File Data")
    print(cal_metrics2(net, test_iter_final, vocab))
    torch.save(net.state_dict(), "encoder_block_model.pt")

elif train_or_infer=="infer": #infer from pretrained weights
    test_net = EncoderBlockModel(vocab_size = len(vocab), embed_size = 100, num_hiddens = 100, num_heads = 5, ffn_num_hiddens = 256, num_layers = 1, dropout = 0.5, num_classes = 2)
    test_iter_final=load_data_imdb2(batch_size=64, train_vocab=vocab, num_steps=500)
    test_net.load_state_dict(torch.load("encoder_block_model.pt", weights_only=True))
    print("Test on CSV File Data")
    print("F1,Precision,Recall,Accuracy")
    print(cal_metrics2(test_net, test_iter_final,vocab))