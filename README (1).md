# Sentiment Analysis on Movie Reviews (PyTorch)

BeeNeural Intern Portal — Data Science & Analytics Task

## Objective
Classify movie reviews as **negative, neutral or positive** using Python and PyTorch.

## Dataset
Stanford Sentiment Treebank (SST-5), loaded with Hugging Face `datasets` (`SetFit/sst5`).
The five original labels are merged into three classes: 0-1 -> negative, 2 -> neutral, 3-4 -> positive.

## Method
1. Lowercase tokenization (words, numbers, `!`, `?`)
2. Vocabulary from the training set only (min frequency 2), padding/truncation to 60 tokens
3. Model: Embedding (128) -> BiLSTM (128 per direction) -> masked mean+max pooling -> Dropout 0.5 -> Linear (3)
4. Training: Adam, class-weighted cross-entropy, gradient clipping, LR scheduler, early stopping on validation accuracy
5. Evaluation on the untouched test set: accuracy, macro F1, per-class report, confusion matrix

## How to run
Open `Sentiment_Analysis_Movie_Reviews.ipynb` in Google Colab, set Runtime -> GPU, and run all cells.
Or locally: `pip install -r requirements.txt`, then run the notebook in Jupyter.

## Results (fill in after running)
| Metric | Value |
|---|---|
| Test accuracy | ___ |
| Macro F1 | ___ |
| Parameters | ___ |
| Training time | ___ |
| Inference speed | ___ reviews/s |

Add `class_distribution.png`, `training_curves.png` and `confusion_matrix.png` (saved by the notebook) to the repository.

## Limitations and future work
Word vectors are learned from scratch on a small dataset; sarcasm and negation remain hard.
Possible improvements: GloVe embeddings, fine-tuning DistilBERT, more data, hyper-parameter search.
