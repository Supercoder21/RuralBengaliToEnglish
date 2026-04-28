## NOTICE: Baseline Transformer Model

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_enc import positional_encoding
from layer_norm import layer_norm

def to_numpy(t):
    return t.detach().cpu().numpy()

def to_torch(a,ref):
    return torch.tensor(a,dtype=ref.dtype, device=ref.device)


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq=100,dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe_np = positional_encoding(max_seq, d_model)
        pe_t=torch.tensor(pe_np, dtype=torch.float32)
        self.register_buffer('pe', pe_t.unsqueeze(0))

    def forward(self,x):
        x = x+self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self,d_model,epsilon=1e-6):
        super().__init__()
        self.epsilon=epsilon
        self.weights = nn.Parameter(torch.ones(d_model))
        self.bias=nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weights, self.bias, self.epsilon)


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__()
        self.n_heads=n_heads
        self.d_k = d_model//n_heads
        self.Wq=nn.Linear(d_model,d_model,bias=False)
        self.Wk=nn.Linear(d_model,d_model,bias=False)
        self.Wv=nn.Linear(d_model,d_model,bias=False)
        self.Wo = nn.Linear(d_model,d_model,bias=False)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, q_in, k_in, v_in, mask=None):
        B, Sq, _ = q_in.shape
        _, Sk, _ = k_in.shape
        Q = self.Wq(q_in).view(B, Sq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(k_in).view(B, Sk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(v_in).view(B, Sk, self.n_heads, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        attn_out = attn_weights @ V
        merged = attn_out.transpose(1, 2).contiguous().view(B, Sq, self.n_heads * self.d_k)
        return self.Wo(merged)


class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.l1=nn.Linear(d_model,d_ff)
        self.l2=nn.Linear(d_ff,d_model)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        return self.l2(self.dropout(self.relu(self.l1(x))))


class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.mha=MultiHeadAttention(d_model,n_heads,dropout)
        self.ffn=FeedForward(d_model,d_ff,dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2=LayerNorm(d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,src_mask=None):
        x=self.ln1(x+self.dropout(self.mha(x,x,x,src_mask)))
        x = self.ln2(x+self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,n_heads,dropout)
        self.cross_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.ffn=FeedForward(d_model,d_ff,dropout)
        self.ln1=LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3=LayerNorm(d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,enc_out,tgt_mask=None,src_mask=None):
        x=self.ln1(x+self.dropout(self.self_attn(x,x,x,tgt_mask)))
        x=self.ln2(x + self.dropout(self.cross_attn(x,enc_out,enc_out,src_mask)))
        x=self.ln3(x+self.dropout(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model=512,n_heads=8,
                 n_layers=6,d_ff=2048,max_seq=100,dropout=0.1,pad_id=0):
        super().__init__()
        self.d_model=d_model
        self.pad_id = pad_id

        self.src_embed=nn.Embedding(src_vocab_size,d_model,padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size,d_model,padding_idx=pad_id)
        self.pos_enc=PositionalEncoding(d_model,max_seq,dropout)

        self.encoder=nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)])
        self.decoder=nn.ModuleList([DecoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)])

        self.output_proj = nn.Linear(d_model,tgt_vocab_size,bias=False)
        self._init_weights()
        self.output_proj.weight=self.tgt_embed.weight

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _padding_mask(self,token_ids):
        return ((token_ids==self.pad_id).float()*-1e9).unsqueeze(1).unsqueeze(2)

    def _causal_mask(self,seq_len,device):
        return torch.triu(torch.full((seq_len,seq_len),-1e9,device=device),diagonal=1).unsqueeze(0).unsqueeze(0)

    def encode(self,src):
        src_mask=self._padding_mask(src)
        x = self.pos_enc(self.src_embed(src)*math.sqrt(self.d_model))
        for layer in self.encoder:
            x=layer(x,src_mask)
        return x,src_mask

    def decode(self,tgt,enc_out,src_mask):
        tgt_mask=self._causal_mask(tgt.size(1),tgt.device)
        x=self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x=layer(x,enc_out,tgt_mask,src_mask)
        return x

    def forward(self,src,tgt):
        enc_out,src_mask=self.encode(src)
        dec_out = self.decode(tgt,enc_out,src_mask)
        return self.output_proj(dec_out)

    @torch.no_grad()
    def greedy_decode(self,src,bos_id,eos_id,max_len=50):
        self.eval()
        enc_out,src_mask=self.encode(src)
        tgt=torch.tensor([[bos_id]],device=src.device)
        generated=[]

        for _ in range(max_len):
            logits=self.output_proj(self.decode(tgt,enc_out,src_mask))
            next_id=logits[0,-1,:].argmax().item()
            if next_id==eos_id:
                break
            generated.append(next_id)
            tgt=torch.cat([tgt,torch.tensor([[next_id]],device=src.device)],dim=1)

        return generated

    def save(self,path,vocab,optimizer=None,scheduler=None):
        checkpoint={
            "model"  : self.state_dict(),
            "vocab"  : vocab,
            "d_model": self.d_model,
            "pad_id" : self.pad_id,
        }
        if optimizer  is not None: checkpoint["optimizer"] =optimizer.state_dict()
        if scheduler  is not None: checkpoint["step_num"]  =scheduler.step_num
        torch.save(checkpoint,path)
        print(f"Saved to {path}")

    @staticmethod
    def load(path,device="cpu"):
        checkpoint=torch.load(path,map_location=device)
        vocab   =checkpoint["vocab"]
        d_model =checkpoint["d_model"]
        pad_id  =checkpoint["pad_id"]
        model=Transformer(
            src_vocab_size=len(vocab),
            tgt_vocab_size=len(vocab),
            d_model=d_model,
            pad_id=pad_id,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded from {path}")
        return model,checkpoint


class LabelSmoothingLoss(nn.Module):
    def __init__(self,vocab_size,pad_id=0,smoothing=0.1):
        super().__init__()
        self.vocab_size=vocab_size
        self.pad_id=pad_id
        self.smoothing=smoothing
        self.confidence = 1.0-smoothing

    def forward(self,logits,targets):
        logits = logits.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)
        log_probs=F.log_softmax(logits,dim=-1)
        smooth_dist=torch.full_like(log_probs,self.smoothing/(self.vocab_size-2))
        smooth_dist.scatter_(1,targets.unsqueeze(1),self.confidence)
        smooth_dist[:,self.pad_id]=0.0
        pad_mask=(targets==self.pad_id)
        smooth_dist[pad_mask]=0.0
        loss=-(smooth_dist*log_probs).sum(dim=-1)
        return loss.sum()/(~pad_mask).sum()


class WarmupScheduler:
    def __init__(self,optimizer,d_model,warmup_steps=4000):
        self.optimizer=optimizer
        self.d_model=d_model
        self.warmup_steps=warmup_steps
        self.step_num=0

    def step(self):
        self.step_num+=1
        lr=(self.d_model**-0.5)*min(self.step_num**-0.5,self.step_num*self.warmup_steps**-1.5)
        for group in self.optimizer.param_groups:
            group['lr']=lr
        return lr
