# Rural Bengali Dialect → English: A Transformer-Based Translation Pipeline

A three-stage sequence-to-sequence pipeline that takes rural Bengali dialect audio all the way to English text. Stage one converts audio to rural Bengali script via ASR. Stage two normalises the dialect to standard Bengali using a custom-trained mBART ensemble. Stage three translates standard Bengali to English. The full pipeline achieves 38.55 BLEU on the Vashantor benchmark.

---

## Setup

```bash
git clone https://github.com/Supercoder21/RuralBengaliToEnglish
cd RuralBengaliToEnglish
pip install -r requirements.txt
```

---

## Data

The text translation stages use the [Vashantor dataset](https://doi.org/10.17632/r5vzd7zx2g.1) (Khandaker et al., 2025), a parallel corpus of 32,500 sentence triplets across five rural Bengali dialects: Barishal, Chittagong, Mymensingh, Noakhali, and Sylhet. Each triplet contains the rural dialect utterance, its standard Bengali equivalent, and an English translation.

Download the train, validation, and test JSON files for each dialect and place them in the root directory. Running the first five cells of the Colab in order produces `vashantor_train.csv`, `vashantor_validation.csv`, and `vashantor_test.csv` which all training and evaluation cells use.

---

## How To Access

**Translation Pipeline**

1. Download the pretrained models from the links below and place them in a folder called `bengali_translation` in your Google Drive
2. Open `training/colab_finetune.ipynb` in Google Colab
3. Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU
4. Mount your Google Drive when prompted
5. Upload `vashantor_train.csv`, `vashantor_validation.csv`, and `vashantor_test.csv` to the Colab session
6. Run the cells based on the models needed and the different strategies.

**ASR Pipeline**

To Be Added by Avnish

---

## Pretrained Models

Download the models and place them inside a folder called `bengali_translation` in your Google Drive. The notebook expects this exact folder name and structure.

**Rural Bengali to Standard Bengali (r2s), ranked by quality**

- [mbart_dacf](https://drive.google.com/drive/folders/1fnl9QLZbHLLj53AnrZT2VKPLdVbsyU5A?usp=sharing) - DACF model alone, combined with curriculum model, 36.09 BLEU
- [mbart_mrasp2_curriculum](https://drive.google.com/drive/folders/1Nk9YqePZgbiF3fct6km4mSHYmYVAoPF1?usp=sharing) - curriculum model alone, 35.42 BLEU
- [mbart_mrasp2_r2s](https://drive.google.com/drive/folders/1JbGAkGRkyM8FYSR6LRe0VwSOQvd-eKR2?usp=sharing) - mRASP2 r2s model, 33.58 BLEU
- [mbart_mrasp2_pretrained](https://drive.google.com/drive/folders/1x3XPvAc31OjOByaEroO2mIfwJJOg6EAq?usp=sharing) - mRASP2 pretraining checkpoint
- [mbart_r2s](https://drive.google.com/drive/folders/12SfwmrvjmgIkfXjcqMBQnEjPH_ds2U64?usp=sharing) - mBART baseline, 31.64 BLEU

**Standard Bengali to English (s2e), ranked by quality on normalized input**

- [mbart_pipeline_s2e](https://drive.google.com/drive/folders/1dozRWSjOKbQNTnPOcD3vGXizJOPWKX12?usp=sharing) - 36.92 BLEU on normalized input — best on noisy r2s output, use checkpoint-879 inside this folder
- [helsinki_s2e](https://drive.google.com/drive/folders/1lk4OpHKx5zooIUsLa5yBz_9CrJkqFGvy?usp=sharing) - 49.47 BLEU on clean input, 34.67 BLEU on normalized input — you need this alongside mbart_pipeline_s2e for the full pipeline

**Other models (experimental, they are unneeded for inference, not in the final model)**

- [mbart_r2e](https://drive.google.com/drive/folders/1MnUKnaMuNkMraCkEQIi_ZzpEV60jB-CG?usp=sharing) - direct rural→English baseline, no normalization stage
- [mbart_s2e](https://drive.google.com/drive/folders/1TN-4mb5qOu0QAo9m_DYMfm2hIc1i3MLv?usp=sharing) - standard mBART s2e baseline
- [mbart_joint](https://drive.google.com/drive/folders/1vRv0oMWJNXLMY_8Fo_ZKrLp8MIPuBb55?usp=sharing) - joint encoder multi-decoder experiment, 29.31 BLEU
- [helsinki_pipeline_s2e](https://drive.google.com/drive/folders/1dNBbrH060fDVzSLXYNanjya9Y0FJWbKw?usp=sharing) - pipeline-aware Helsinki experiment
- [mbart_pass2_refine](https://drive.google.com/drive/folders/1CljHFNrffi2rNwDuuAVZKVpl7LQ6IdMm?usp=sharing) - pass 2 refinement experiment
- [qe_reranker_v2](https://drive.google.com/drive/folders/14x7yQiKX6JPliWANRtwHL5U7kjHHkK1c?usp=sharing) - pairwise quality estimation reranker
- [reranker_model](https://drive.google.com/drive/folders/10umodekR50OqeCC7FuW8gGKGK-bpROA0?usp=sharing) - margin reranker experiment
- [rl_selector](https://drive.google.com/drive/folders/1wjsGulW-LIfx0B-MmYB1_vpANvYddF65?usp=sharing) - reinforcement learning candidate selector
- [qwen_r2s_v3_sft_best](https://drive.google.com/drive/folders/1TUGKwomTRBASCKJa08WxSGo_pahWZxKb?usp=sharing) - best Qwen2.5-7B LoRA r2s attempt, 27.52 BLEU

**How the models are combined**

No single model file produces the final result on its own, the pipeline chains everything through code. At the r2s stage the DACF and curriculum models are logit-averaged during beam search, blending their token probability distributions at every decoding step into a single normalized Bengali output. At the s2e stage both Helsinki and the pipeline-aware mBART generate candidates, beam scores are compared, the more confident model's output is selected as the base, and span-level fusion is applied to refine it further by swapping word spans that score better under the model's own loss. The inference cell at the bottom of the notebook does all of this end to end. It loads the models and it handles the rest.

**ASR Pipeline**

- [ASR_Pipeline] - to be added by Avnish

**How the model is integrated with r2s2e**

To Be Added
---

## Results

Full r2s2e pipeline on the Vashantor test set:

- r2s DACF + Curriculum ensemble - 36.74 BLEU, 66.68 chrF
- s2e pipeline-aware mBART on normalized input - 36.92 BLEU, 55.69 chrF
- s2e beam vote ensemble - 38.23 BLEU, 57.66 chrF
- **Full pipeline with span fusion - 38.55 BLEU, 57.83 chrF**

---

## Linguistic Divergence

A lingustic study between rural Bengali and standard Bengali was also conducted in an effort to learn how similar different dialects were from their standard versions. The results, which include word-embeddings, hub-words, and alignment coefficients, are as followed:

To Be Added by Aarav and Avnish

---

## Archive

The `training/scratch_transformer_baseline/` folder contains an earlier attempt at training a transformer encoder-decoder from scratch on the romanised rural-to-standard task. It achieved a best validation loss of 1.5951 at epoch 35 but produced noisy outputs. 44M parameters was simply too large for our previous 4,653-pair corpus. The final pipeline uses fine-tuned pretrained models instead.

---

## Authors

**Aarav** - all of r2s and s2e translation pipeline: model training, ensemble methods, reranking experiments and system combination, ablation results | lingustic divergence analysis: word embeddings, cosine similarity

**Avnish** - all of ASR pipeline | linguistic divergence analysis: word embeddings + alignment, n-nearest neighbors, hub words

---

## Citation

If you use this work, please cite the Vashantor dataset:

```
Faria, F. T. J., et al. (2024). Vashantor: A Large-scale Multilingual Benchmark Dataset.
Mendeley Data. https://doi.org/10.17632/r5vzd7zx2g.1
```



