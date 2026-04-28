## Archived Files

This folder contains an earlier attempt at training a transformer encoder-decoder from scratch on the romanized rural-to-standard task. This model achieved a best validation loss of 1.5951 at epoch 35 but produced noisy outputs due to the small corpus size (4,653 pairs) relative to the model's 44M parameters. The final pipeline uses fine-tuned pretrained models instead — see the `training/` folder for the current fine-tuning scripts.

The from-scratch implementation is preserved here as a baseline and for documentation 
of the architectural work described in the paper.
