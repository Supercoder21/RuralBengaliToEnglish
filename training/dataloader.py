## NOTICE: Used in primitive transformer implementation
import torch

PAD,BOS,EOS,UNK = "<PAD>","<BOS>","<EOS>","<UNK>"

def load_pairs(filepath):
    pairs=[]
    with open(filepath,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if "|||" in line:
                src,tgt=line.split("|||",1)
                pairs.append((src.strip(),tgt.strip()))
    return pairs


class CharTokenizer:
    def __init__(self,pairs):
        chars=set()
        for s,t in pairs:
            chars.update(s)
            chars.update(t)
        self.vocab=[PAD,BOS,EOS,UNK]+sorted(chars)
        self.ch2id={c:i for i,c in enumerate(self.vocab)}
        self.id2ch={i:c for c,i in self.ch2id.items()}
        self.pad_id=self.ch2id[PAD]
        self.bos_id=self.ch2id[BOS]
        self.eos_id=self.ch2id[EOS]
        self.unk_id=self.ch2id[UNK]

    @property
    def vocab_size(self): return len(self.vocab)

    def encode(self,text,add_bos=False,add_eos=True):
        ids=[self.ch2id.get(c,self.unk_id) for c in text]
        if add_bos: ids=[self.bos_id]+ids
        if add_eos: ids=ids+[self.eos_id]
        return ids

    def decode(self,ids):
        return "".join(self.id2ch.get(i,UNK) for i in ids
                       if self.id2ch.get(i,UNK) not in (PAD,BOS,EOS,UNK))

    def pad(self,ids,max_len):
        ids=ids[:max_len]
        return ids+[self.pad_id]*(max_len-len(ids))


def make_batch(pairs,tok,max_len=100,device="cpu"):
    src_l,ti_l,to_l=[],[],[]
    for s,t in pairs:
        s_ids=tok.pad(tok.encode(s,add_eos=True),max_len)
        t_ids=tok.encode(t,add_eos=True)
        ti=tok.pad([tok.bos_id]+t_ids[:-1],max_len)
        to=tok.pad(t_ids,max_len)
        src_l.append(s_ids)
        ti_l.append(ti)
        to_l.append(to)
    return (
        torch.tensor(src_l, dtype=torch.long,device=device),
        torch.tensor(ti_l,  dtype=torch.long,device=device),
        torch.tensor(to_l,  dtype=torch.long,device=device),
    )


def make_batches(pairs,tok,max_len=100,batch_size=32,device="cpu"):
    batches=[]
    for i in range(0,len(pairs),batch_size):
        batch=pairs[i:i+batch_size]
        batches.append(make_batch(batch,tok,max_len,device))
    return batches
