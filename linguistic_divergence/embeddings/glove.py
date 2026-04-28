import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


def build_vocab(filepath,min_count=1):
    counts=defaultdict(int)
    with open(filepath,"r",encoding="utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counts[word]+=1
    vocab=[w for w,c in counts.items() if c>=min_count]
    w2i={w:i for i,w in enumerate(vocab)}
    i2w={i:w for w,i in w2i.items()}
    return w2i,i2w


def build_cooccurrence(filepath,w2i,window=5):
    cooc=defaultdict(float)
    with open(filepath,"r",encoding="utf-8") as f:
        for line in f:
            words=[w for w in line.strip().split() if w in w2i]
            for i,word in enumerate(words):
                wi=w2i[word]
                for j in range(max(0,i-window),min(len(words),i+window+1)):
                    if i==j: continue
                    wj=w2i[words[j]]
                    cooc[(wi,wj)]+=1.0/abs(i-j)
    return cooc


def weighting_fn(x,x_max=100,alpha=0.75):
    return torch.where(x<x_max,(x/x_max)**alpha,torch.ones_like(x))


class GloVe(nn.Module):
    def __init__(self,vocab_size,embed_dim=50):
        super().__init__()
        self.W  =nn.Embedding(vocab_size,embed_dim)
        self.Wc =nn.Embedding(vocab_size,embed_dim)
        self.b  =nn.Embedding(vocab_size,1)
        self.bc =nn.Embedding(vocab_size,1)
        nn.init.uniform_(self.W.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.uniform_(self.Wc.weight,-0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.bc.weight)

    def forward(self,wi,wj,xij):
        dot=(self.W(wi)*self.Wc(wj)).sum(dim=1)
        bi =self.b(wi).squeeze()
        bj =self.bc(wj).squeeze()
        diff=dot+bi+bj-torch.log(xij.clamp(min=1e-10))
        loss=(weighting_fn(xij)*(diff**2)).mean()
        return loss

    def get_vectors(self):
        W =self.W.weight.detach().cpu().numpy()
        Wc=self.Wc.weight.detach().cpu().numpy()
        return (W+Wc)/2.0


def train_glove(filepath,embed_dim=50,window=5,epochs=50,batch_size=512,lr=0.05):
    print(f"\nTraining GloVe on: {filepath}")
    w2i,i2w=build_vocab(filepath)
    print(f"Vocab size : {len(w2i)}")

    print("Building co-occurrence matrix...")
    cooc=build_cooccurrence(filepath,w2i,window)
    print(f"Co-occurrence pairs: {len(cooc)}")

    pairs=list(cooc.items())
    wi_all =np.array([p[0][0] for p in pairs],dtype=np.int64)
    wj_all =np.array([p[0][1] for p in pairs],dtype=np.int64)
    xij_all=np.array([p[1]    for p in pairs],dtype=np.float32)

    model=GloVe(len(w2i),embed_dim)
    optimizer=torch.optim.Adagrad(model.parameters(),lr=lr)
    n=len(pairs)

    for epoch in range(1,epochs+1):
        idx=np.random.permutation(n)
        wi_all=wi_all[idx]; wj_all=wj_all[idx]; xij_all=xij_all[idx]
        total_loss=0.0; steps=0
        for start in range(0,n,batch_size):
            wi =torch.tensor(wi_all[start:start+batch_size], dtype=torch.long)
            wj =torch.tensor(wj_all[start:start+batch_size], dtype=torch.long)
            xij=torch.tensor(xij_all[start:start+batch_size],dtype=torch.float32)
            optimizer.zero_grad()
            loss=model(wi,wj,xij)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item(); steps+=1
        if epoch%10==0:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss/steps:.4f}")

    return model,w2i,i2w


def save_vectors(model,w2i,i2w,outpath):
    vecs=model.get_vectors()
    np.save(outpath+".npy",vecs)
    with open(outpath+".txt","w",encoding="utf-8") as f:
        f.write(f"{len(w2i)} {vecs.shape[1]}\n")
        for i,w in i2w.items():
            vec=" ".join(f"{v:.6f}" for v in vecs[i])
            f.write(f"{w} {vec}\n")
    print(f"Saved to {outpath}.npy and {outpath}.txt")
    return vecs


def load_vectors(path):
    vecs=np.load(path+".npy")
    w2i={}; i2w={}
    with open(path+".txt","r",encoding="utf-8") as f:
        f.readline()  # skip header
        for i,line in enumerate(f):
            word=line.strip().split()[0]
            w2i[word]=i; i2w[i]=word
    return vecs,w2i,i2w


def nearest_neighbours(word,vecs,w2i,i2w,top_n=5):
    if word not in w2i:
        print(f"'{word}' not in vocab"); return []

    idx=w2i[word]
    query=vecs[idx]

    # cosine similarity against all vectors
    norms=np.linalg.norm(vecs,axis=1,keepdims=True)
    normed=vecs/(norms+1e-10)
    query_normed=query/(np.linalg.norm(query)+1e-10)
    sims=normed@query_normed

    sims[idx]=-1  # exclude the word itself
    top=np.argsort(sims)[::-1][:top_n]
    results=[(i2w[i],float(sims[i])) for i in top]
    return results


if __name__=="__main__":
    # train and save both

    # replace with the path of wherever both files are saved
    model_a,w2i_a,i2w_a=train_glove("rural_bengali.txt",embed_dim=50,epochs=50)  
    vecs_a=save_vectors(model_a,w2i_a,i2w_a,"standard_bengali.txt")

    model_b,w2i_b,i2w_b=train_glove("corpus_b.txt",embed_dim=50,epochs=50)
    vecs_b=save_vectors(model_b,w2i_b,i2w_b,"glove_standard")

    # nearest neighbour demo
    print("\n── Rural Bengali nearest neighbours ──")
    test_words_a=list(w2i_a.keys())[:10]  # just grab first 10 words as demo
    for w in test_words_a:
        neighbours=nearest_neighbours(w,vecs_a,w2i_a,i2w_a,top_n=5)
        print(f"\n  '{w}':")
        for neighbour,sim in neighbours:
            print(f"    {neighbour:20s}  {sim:.4f}")

    print("\n── Standard Bengali nearest neighbours ──")
    test_words_b=list(w2i_b.keys())[:10]
    for w in test_words_b:
        neighbours=nearest_neighbours(w,vecs_b,w2i_b,i2w_b,top_n=5)
        print(f"\n  '{w}':")
        for neighbour,sim in neighbours:
            print(f"    {neighbour:20s}  {sim:.4f}")
