import numpy as np
import pickle
import json
from pathlib import Path
from collections import Counter
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from glove_train import load_vectors, nearest_neighbours

Path("results").mkdir(exist_ok=True)

vecs_r, w2i_r, i2w_r = load_vectors("glove_rural")
vecs_s, w2i_s, i2w_s = load_vectors("glove_standard")

shared_vocab = sorted(set(w2i_r) & set(w2i_s))
print(f"Rural vocab:    {len(w2i_r)}")
print(f"Standard vocab: {len(w2i_s)}")
print(f"Shared vocab:   {len(shared_vocab)}")

X = np.array([vecs_r[w2i_r[w]] for w in shared_vocab], dtype=np.float32)
Y = np.array([vecs_s[w2i_s[w]] for w in shared_vocab], dtype=np.float32)

X = X / np.linalg.norm(X, axis=1, keepdims=True)
Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)

R, _       = orthogonal_procrustes(X, Y)
X_aligned  = X @ R

frob_before    = np.linalg.norm(X - Y,         ord="fro")
frob_after     = np.linalg.norm(X_aligned - Y, ord="fro")
norm_disparity = frob_after / len(shared_vocab)

print(f"\nFrobenius before alignment: {frob_before:.4f}")
print(f"Frobenius after alignment:  {frob_after:.4f}")
print(f"Normalised disparity:       {norm_disparity:.6f}")

diag_sims  = np.einsum("ij,ij->i", X_aligned, Y)
diag_dists = 1.0 - diag_sims

mean_dist   = float(np.mean(diag_dists))
median_dist = float(np.median(diag_dists))
std_dist    = float(np.std(diag_dists))

print(f"\nMean cosine distance:   {mean_dist:.4f}")
print(f"Median cosine distance: {median_dist:.4f}")
print(f"Std cosine distance:    {std_dist:.4f}")

BATCH      = 512
nn_indices = np.empty(len(shared_vocab), dtype=int)
for start in range(0, len(shared_vocab), BATCH):
    end                     = min(start + BATCH, len(shared_vocab))
    sims                    = cosine_similarity(X_aligned[start:end], Y)
    nn_indices[start:end]   = np.argmax(sims, axis=1)

hits   = int(np.sum(nn_indices == np.arange(len(shared_vocab))))
nn_acc = hits / len(shared_vocab)
print(f"\nNN retrieval accuracy (P@1): {nn_acc:.4f}  ({hits}/{len(shared_vocab)})")

hub_counts    = Counter(int(i) for i in nn_indices)
top_hubs      = [(shared_vocab[idx], cnt) for idx, cnt in hub_counts.most_common(20)]

print("\nTop 20 hub words (standard space):")
for word, cnt in top_hubs:
    print(f"  {word:25s}  {cnt}")

sorted_idx       = np.argsort(diag_dists)[::-1]
top_divergent    = [(shared_vocab[i], float(diag_dists[i])) for i in sorted_idx[:30]]
top_stable       = [(shared_vocab[i], float(diag_dists[i])) for i in sorted_idx[-30:]]

print("\nTop 30 most divergent words:")
for w, d in top_divergent:
    print(f"  {w:25s}  {d:.4f}")

print("\nTop 30 most stable words:")
for w, d in top_stable:
    print(f"  {w:25s}  {d:.4f}")

results = {
    "global": {
        "shared_vocab_size":  len(shared_vocab),
        "mean_cosine_dist":   mean_dist,
        "median_cosine_dist": median_dist,
        "std_cosine_dist":    std_dist,
        "nn_accuracy_p1":     nn_acc,
        "nn_hits":            hits,
        "frob_before":        float(frob_before),
        "frob_after":         float(frob_after),
        "norm_disparity":     float(norm_disparity),
    },
    "top_divergent": top_divergent,
    "top_stable":    top_stable,
    "top_hubs":      top_hubs,
    "word_distances": {shared_vocab[i]: float(diag_dists[i]) for i in range(len(shared_vocab))},
}

with open("results/divergence_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

np.save("results/nn_indices.npy", nn_indices)
np.save("results/diag_dists.npy", diag_dists)
np.save("results/X_aligned.npy",  X_aligned)

print("\nSaved → results/divergence_results.json")
