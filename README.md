# Rural Bengali Dialect → English: A Transformer-Based Translation Pipeline

A sequence-to-sequence pipeline for transliterating rural Bengali dialect into English via three stages: audio → rural Bengali, rural → standard Bengali dialect normalization, and then standard Bengali → English translation.

---

## Setup

```
git clone https://github.com/Supercoder21/RuralBengaliToEnglish
cd RuralBengaliToEnglish
pip install -r requirements.txt
```

---

## Data

The corpus is sourced from the BanglaRegionalTextCorpus (Ahmed et al., 2026), available at https://doi.org/10.1016/j.dib.2026.001381. Download the Excel file and place it in the root directory. It is not included in this repo.

---

## Training

### Recommended: Google Colab (GPU)

1. Open 'training/colab_finetune.ipynb' in Google Colab
2. Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Upload 'corpus_aligned.txt' and 'BanglaRegionalTextCorpus-tnQRMn.xlsx' when prompted
5. All models save automatically to your Google Drive under 'bengali_translation/'

### Local (CPU only — slow)

Code To Be Added:
```
python training/finetune_rural_to_standard.py
python training/finetune_standard_to_english.py
```

---

## Pretrained Models

The following trained models are available: 


Rural Bengali Audio → Rural Bengali Dialect:

Rural Bengali → Standardized:

Standardized Bengali → English:
https://drive.google.com/drive/folders/1TN-4mb5qOu0QAo9m_DYMfm2hIc1i3MLv?usp=drive_link


Download and place the folders in the root directory to use them for inference.

---

## Archive

The `training/scratch_transformer_baseline/` folder contains an earlier attempt at training a transformer encoder-decoder from scratch on the romanized rural-to-standard task. This model achieved a best validation loss of 1.5951 at epoch 35 but produced noisy outputs due to the small corpus size (4,653 pairs) relative to the model's 44M parameters. The final pipeline uses fine-tuned pretrained models instead. The from-scratch implementation is preserved as a baseline and for documentation of the architectural work described in the paper.

---

## Citation

If you use this work, please cite:

Ahmed et al. (2026). BanglaRegionalTextCorpus: A Curated Dataset for Four Regional Bangla Dialects. Data in Brief. https://doi.org/10.1016/j.dib.2026.001381
