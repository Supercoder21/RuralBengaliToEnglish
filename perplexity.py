## NOTICE: Unused example of Perplexity

import math
def compute_perplexity(sentence, bigram_probs):
    words = ["<s>"] + sentence.split()
    log_sum = 0
    T = len(words) - 1
    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]

        prob = bigram_probs.get((prev_word, curr_word))
        print('Probability for P['+prev_word+','+curr_word+'] = '+str(prob))
        prior_log_sum = log_sum
        log_sum += math.log(prob)
        print('Sum is now:',prior_log_sum,'+ log('+str(prob)+') =',log_sum)

    perplexity = math.exp(-log_sum / T)
    print()
    print('Perplexity is exp(-1 * '+str(log_sum)+'/'+str(T)+') =',perplexity)
    return perplexity

sentence = "the cat sat on the mat"
bigram_probs = {("<s>", "the"): 0.5,("the", "cat"): 0.4,("cat", "sat"): 0.3,("sat", "on"): 0.6,("on", "the"): 0.5,("the", "mat"): 0.2,}

pp = compute_perplexity(sentence, bigram_probs)
