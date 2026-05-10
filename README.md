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
The corpus used for the text translation task is the Vashantor dataset (Khandaker et al., 2025). Download the train, validation and test JSON files for each dialect (Barishal, Chittagong, Mymensingh, Noakhali, Sylhet) and place them in the root directory.
Running the first five cells in order in the Google Colab gives you the files vashantor_train.csv, vashantor_validation.csv, and vashantor_test.csv used for all training and evaluation.

---

## How To Access

### Recommended: Google Colab (GPU)

Translation Pipeline:

Firstly, put translation models into Google Drive in the bengali_translation folder. 

Next:
1. Open 'training/colab_finetune.ipynb' in Google Colab
2. Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Upload Vashantor Training, Testing, and Validation translation sets to the Drive.
5. Models can be accesed and tested from there on out.

ASR Pipeline:
[TO BE ADDED]
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
https://drive.google.com/drive/folders/12SfwmrvjmgIkfXjcqMBQnEjPH_ds2U64?usp=drive_link

Standardized Bengali → English:
Available directly in the Google Colab


Download and place the model files in the "bengali_translation" folder to use them for inference.

---

## Archive

The `training/scratch_transformer_baseline/` folder contains an earlier attempt at training a transformer encoder-decoder from scratch on the romanized rural-to-standard task. This model achieved a best validation loss of 1.5951 at epoch 35 but produced noisy outputs due to the small corpus size (4,653 pairs) relative to the model's 44M parameters. The final pipeline uses fine-tuned pretrained models instead. The from-scratch implementation is preserved as a baseline and for documentation of the architectural work described in the paper.

---

## Citation

If you use this work, please cite:

Faria, F. T. J., et al. (2024). Vashantor: A Large-scale Multilingual Benchmark Dataset [Dataset]. Mendeley Data. doi.org
