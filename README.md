# Rural Bengali Dialect → English: A Transformer-Based Translation Pipeline

A three-stage sequence-to-sequence pipeline that takes rural Bengali dialect audio all the way to English text. Stage one converts audio to rural Bengali script via ASR. Stage two normalises the dialect to standard Bengali using a custom-trained mBART ensemble. Stage three translates standard Bengali to English. The r2s2e pipeline achieves 38.55 BLEU on the Vashantor benchmark. The full pipeline, with ASR, achieves (Avnish to Add After Integration) on the training set.

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


An end-to-end system for translating rural Bengali dialect speech into English, comprising ASR, dialect normalisation, and cross-lingual translation.
Pipeline
Audio (WAV/MP3)  →  Whisper ASR  →  Bengali Script  →  Rural→Standard  →  Standard→English

     ↑                  ↑                                    ↑                    ↑

  yt-dlp          Whisper large-v3                     DACF+Curriculum      Helsinki-NLP +

  16kHz mono       language=bn                          mBART-50 ensemble   Pipeline-aware mBART
Repository Structure
RuralBengaliToEnglish/

├── data/

│   ├── corpus_A/corpus_a.txt           # 874 sentences, Purulia dialect

│   └── corpus_B/corpus_b.txt           # 9,993 sentences, standard Bengali

├── embeddings/

│   ├── sgns.py                         # SGNS from scratch (Avnish)

│   ├── data_utils.py                   # Vocab, encoding, alias sampler

│   └── sgns_embeddings_*.npy           # Trained embeddings (joint, DA, DB)

├── alignment/

│   └── eval_divergence.py              # Procrustes + JS divergence analysis

├── metrics/

│   ├── levenshtein.py                  # Levenshtein distance (Avnish)

│   └── bleu.py                         # BLEU-1 to BLEU-4 (Avnish)

├── model/

│   └── attention.py                    # Scaled dot-product attention (Avnish)

├── scripts/

│   ├── asr_pipeline.py                 # Audio → Whisper → Bengali script

│   └── scrape_dialect_audio.py         # YouTube dialect audio scraper

├── figures/                            # All plots and visualisations

├── requirements.txt

└── README.md
ASR Component
Overview
The ASR subsystem transcribes rural Bengali dialect audio into Bengali script using OpenAI's Whisper large-v3 (1.55B parameters). Whisper outputs Bengali script directly, which is passed unmodified to the downstream normalisation and translation models.
Audio Acquisition
Rural Bengali dialect audio was sourced from YouTube using yt-dlp. Eight long-form Purulia/Manbhumi dialect videos (folk interviews, village conversations, local cultural programming) totalling 296 minutes (~5 hours) were downloaded and converted to 16kHz mono WAV.

python scripts/scrape_dialect_audio.py --url-file urls.txt --output-dir data/dialect_audio
Whisper Inference
Whisper large-v3 was applied off-the-shelf with language="bn" and task="transcribe". No fine-tuning was performed — Whisper's pre-training includes Bengali speech data, and no ground-truth dialect transcriptions exist for supervised fine-tuning.

python scripts/asr_pipeline.py \

    --da-audio-dir data/dialect_audio \

    --output-dir data \

    --whisper-model large-v3

Output: Bengali script transcriptions (one sentence per line). After deduplication and quality filtering: 874 sentences from 1,006 raw.
Standard Bengali Corpus
OpenSLR-53 (~196K utterances, CC BY-SA 4.0) provides standard Bengali transcriptions. Subsampled to 10,000 sentences.

python scripts/asr_pipeline.py --output-dir data --skip-da
Trained Models
Rural → Standard Bengali (DACF+Curriculum ensemble): Google Drive
BLEU: 36.74 | chrF: 66.68 (+5.10 BLEU over mBART-50 baseline)
Standard → English (Pipeline-aware + span fusion): BLEU: 38.55


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

Linguistic Divergence Analysis
For the divergence analysis only (not the main pipeline), Bengali script is transliterated to ITRANS romanisation for SGNS/GloVe embedding training.

Metric
Result
Shared vocabulary (freq≥3)
3,022 types
Neighbourhood overlap (DA vs DB)
0.026 mean (70% zero overlap)
JS divergence (τ=10)
Bimodal: ~700 words at JS≈0.69, ~300 at JS≈0.3
Procrustes anchor Δ
0.636 (alignment quality poor due to DA corpus size)


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



