## NOTICE: Used in primitive transformer implementation
## Val Score: 1.56 
## Noisy results

import torch
import random
import torch.optim as optim
from transformer_complete import Transformer, LabelSmoothingLoss, WarmupScheduler
from dataloader import load_pairs, CharTokenizer, make_batches

pairs = load_pairs("corpus_aligned.txt")
random.shuffle(pairs)

split       = int(0.9 * len(pairs))
train_pairs = pairs[:split]
val_pairs   = pairs[split:]

device = "cuda" if torch.cuda.is_available() else "cpu"
tok    = CharTokenizer(pairs)

print(f"Train pairs : {len(train_pairs)}")
print(f"Val pairs   : {len(val_pairs)}")
print(f"Vocab size  : {tok.vocab_size}")

train_batches = make_batches(train_pairs,tok,max_len=100,batch_size=32,device=device)
val_batches   = make_batches(val_pairs,  tok,max_len=100,batch_size=32,device=device)

model = Transformer(
    src_vocab_size=tok.vocab_size,
    tgt_vocab_size=tok.vocab_size,
    d_model=512,n_heads=8,n_layers=6,
    d_ff=2048,max_seq=100,dropout=0.1,pad_id=tok.pad_id
).to(device)

criterion = LabelSmoothingLoss(tok.vocab_size,tok.pad_id)
optimizer = optim.Adam(model.parameters(),lr=0,betas=(0.9,0.98),eps=1e-9)
scheduler = WarmupScheduler(optimizer,d_model=512,warmup_steps=4000)

print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}\n")

best_val_loss = float('inf')
no_improve    = 0
patience      = 5

for epoch in range(1,51):
    model.train()
    total_loss=0
    for i,(src,tgt_in,tgt_out) in enumerate(train_batches):
        logits=model(src,tgt_in)
        loss=criterion(logits.view(-1,tok.vocab_size),tgt_out.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()
        total_loss+=loss.item()
        if i%5==0:
            print(f"  Epoch {epoch:3d} | Batch {i}/{len(train_batches)} | Loss: {loss.item():.4f}")

    avg_train = total_loss/len(train_batches)

    model.eval()
    val_loss=0
    with torch.no_grad():
        for src,tgt_in,tgt_out in val_batches:
            logits=model(src,tgt_in)
            loss=criterion(logits.view(-1,tok.vocab_size),tgt_out.view(-1))
            val_loss+=loss.item()
    avg_val = val_loss/len(val_batches)

    print(f"Epoch {epoch:3d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        no_improve    = 0
        model.save("rural_to_standard_best.pt",tok.vocab,optimizer,scheduler)
        print(f"  --> New best saved! (val loss: {best_val_loss:.4f})")
    else:
        no_improve+=1
        print(f"  --> No improvement ({no_improve}/{patience})")
        if no_improve>=patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch%1==0:
        model.save(f"rural_to_standard_epoch{epoch}.pt",tok.vocab,optimizer,scheduler)

model.save("rural_to_standard_final.pt",tok.vocab,optimizer,scheduler)

def translate(text):
    ids=tok.pad(tok.encode(text,add_eos=True),100)
    src=torch.tensor([ids],dtype=torch.long,device=device)
    pred=model.greedy_decode(src,tok.bos_id,tok.eos_id)
    return tok.decode(pred)

print("\n── Test translations ──")
for src,tgt in val_pairs[:5]:
    print(f"  Rural:    {src}")
    print(f"  Expected: {tgt}")
    print(f"  Got:      {translate(src)}\n")
