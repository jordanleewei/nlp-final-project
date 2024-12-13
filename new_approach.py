import math
import torch
from torch import nn
from d2l import torch as d2l


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

