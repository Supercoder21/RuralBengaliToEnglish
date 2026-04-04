import math
def compute_perplexity(sentence, bigram_probs):
    words =["<s>"] + sentence.split()
    log_sum=0
    T=len(words) - 1
    for i in range(1, len(words)):
        prev_word= words[i - 1]
        curr_word=words[i]

        prob=bigram_probs.get((prev_word, curr_word))
        prior_log_sum =log_sum
        log_sum +=math.log(prob)

    perplexity= math.exp(-log_sum / T)
    return perplexity
