# LSTM-Seq2Seq-English-to-Russian-Translation
A group project for my Text Mining class in which I created an LSTM using TensorFlow and compare how it performs to a  Transformer model built with PyTorch. TextMining_Project_Q1_Part1.ipynb was written by me, however the PyTorch Transformer in TextMining_Project_Q1_Part2.ipynb was written by another group member. Part 2 is still included in this repo as I perfromed the model evaluation at the end of the Part2 markdown file. 

Here is a brief summary of what I learned/did throughout TextMining_Project_Q1_Part1

## Setup & Data Loading

* Verified GPU/CUDA availability for TensorFlow
* Loaded an English-Russian sentence pair dataset from Tatoeba (~787K pairs) into a DataFrame

## Data Preparation

* Dropped the ID columns (eng_id, rus_id) as carried no semantic value for translation
* Split the data into train (70%), validation (15%), and test (15%) sets

## Tokenization

* Created separate Keras Tokenizer objects for English and Russian
  * Fit only training data to handle words that are outside of word dictionary (new words model has never seen, words that never appeared in the training set)
* Converted sentences to integer sequences using the trained tokenizers
* Applied the same tokenizers to validation and test sets, using <OOV> tokens to handle unseen words

## Padding

* Analyzed sequence length distributions (mean, std) for both languages
* Set the max padding length to mean + 4 × std (~17 tokens) to avoid excessive zeros while retaining most data points

## Model Building & Training

* Built a Sequential LSTM model: Embedding → LSTM → Bidirectional LSTM → LSTM → Dense (ReLU) → Dense (Softmax)
* Trained for 5 epochs using Adam optimizer and sparse categorical crossentropy, reaching ~74.7% training accuracy
* Saved the trained model as my_model_v2.keras

## Prediction & Evaluation

* Generated predictions on the first 100 test samples (full test set caused GPU OOM)
  * This error gave insight into the inner working of how Tensorflow uses data and how GPUs/CPUs manipulate data.
* Decoded predicted token indices back to Russian text
* Computed BLEU scores comparing predicted vs. actual Russian translations
* Observed translation quality and exported results to later compare to transformer model

## Pytorch Model Creation
*skipping not my work*

## Evaluation (Part2 Summary)

### Setup & Comparison Framework

Loaded the LSTM results CSV and added empty columns for Transformer_Translation and Blue_Score_Transformer
Iterated through all 100 English sentences, cleaned each one, and ran them through the transformer's greedy decode function to generate Russian translations

### BLEU Scoring

Scores were computed by comparing each transformer translation against the actual Russian reference sentence

### Bottom Performing Translations

Many failures were caused by <unk> (unknown) tokens dominating the output, especially for specialized or less common vocabulary
Sentences with domain-specific words (e.g. "rear-view mirror", "senate", "mandolin") resulted in largely <unk>-filled translations with BLEU scores of 0

### Key Takeaways

The transformer vastly outperformed the LSTM in translation quality
BLEU scores somewhat underrepresent the transformer's true quality due to punctuation differences and <unk> tokens
The main bottleneck is vocabulary size — expanding the training data would reduce <unk> occurrences significantly
A suggested improvement is to use <unk> tokens as context clues for inference rather than predicting them directly as output words

## Proposed improvements
* Apply the same pre-processing steps that allowed for LSTM model to predict unknown words by using surrounding words as context. avoiding translation with <unk> tags.
* Research and improve how to build an LSTM model properly and understand underlying Deep Learning concepts and tensorflow library
  * Implement encoder and decoder (this is extremely crucial for computation speed and complexity and was missed)
  * Add attenton mechanism (this will greatly improve model predictions as the model now has access to context of previous words)
  * Add more epochs (bias variance trade off, currently only 5 epochs exists)
* If memory is running out on tensorflow .predict() method is causing OOM errors use varied batch sizes to reduce memory strain

Overall I enjoyed this project and look forward to re-visiting it in the future to fix the many issues present.
