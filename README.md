# word2mat

*Word2Mat* is a framework that learns *sentence embeddings* in a CBOW-word2vec style, but where the words and sentences are represented as matrices.
Details of this method and results can be found in our [ICLR paper](https://openreview.net/forum?id=H1MgjoR9tQ).

## Dependencies

- Python3
- PyTorch >= 0.4 with CUDA support
- NLTK >= 3

## Setup python3 environment

Please install the python3 dependencies in your environment:

```
virtualenv -p python3 venv && source venv/bin/activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt')"
```

## Download training data

In order to reproduce the results from our paper, which were trained on the UMBC corpus,
download the [UMBC corpus](https://ebiquity.umbc.edu/resource/html/id/351), extract the tar.gz file,
and run the extract_umbc.py script in the following way:

```
python extract_umbc.py umbc_corpus/webbase_all <path_to_store_sentences>
```

This stores the sentences from the UMBC corpus in a format that is usable by our code: Each line in the resulting file contains a single sentence,
whose (already pre-processed) tokens are separated by a whitespace character.

## Running the experiments

Note: After further experiments, we observed that terminating training based on the validation loss produces unreliable results because of
relatively high variance in the validation loss. Hence, we recommend using training loss as stopping criterion, which is more stable.

The results below are trained with this stopping criterion, and therefore slightly differ from the results reported in the ICLR paper.
However, the conclusions remain the same: CMOW is much better than CBOW at capturing linguistic properties except WordContent.
Therefore, CBOW is superior in almost all downstream tasks except TREC.
The Hybrid model retains the capabilities of both models and therefore is extremely close to the better model among CBOW and CMOW, or better
on all tasks.

Probing tasks: All scores denote accuracy.

| Model |  Depth|  BigramShift|  SubjNumber|  Tense|  CoordinationInversion|  Length|  ObjNumber|  TopConstituents|  OddManOut|  WordContent|
|:----------|------:|------------:|-----------:|------:|----------------------:|-------:|----------:|----------------:|----------:|------------:|
| CBOW      |  32.73|        49.65|       79.65|  79.46|                  53.78|   75.69|      79.00|            72.26|      49.64|        89.11|
| CMOW      |  34.40|        72.44|       82.08|  80.32|                  62.05|   82.93|      79.70|            74.25|      51.33|        65.15|
| Hybrid    |  35.38|        71.22|       81.45|  80.83|                  59.17|   87.00|      79.37|            72.88|      50.53|        86.97|

Supervised downstream tasks: For STS-Benchmark and Sick-Relatedness, the results denote Spearman correlation coefficient. For all others the score denotes accuracy.

| Model |   SNLI|   SUBJ|     CR|     MR|   MPQA|  TREC|  SICKEntailment|   SST2|   SST5|   MRPC|  STSBenchmark|  SICKRelatedness|
|:----------|------:|------:|------:|------:|------:|-----:|---------------:|------:|------:|------:|-------------:|----------------:|
| CBOW      |  67.76|  90.45|  79.76|  74.32|  87.23|  84.4|           79.58|  78.14|  41.72|  72.17|         0.619|            0.721|
| CMOW      |  64.77|  87.11|  74.60|  71.42|  87.55|  88.0|           76.90|  76.77|  40.18|  70.61|         0.576|            0.705|
| Hybrid    |  67.59|  90.26|  79.60|  74.10|  87.38|  89.2|           78.69|  77.87|  41.58|  71.94|         0.613|            0.718|

Unsupervised downstream tasks: The score denotes Spearman correlation coefficient.

| Model |  STS12|  STS13|  STS14|  STS15|  STS16|
|:----------|------:|------:|------:|------:|------:|
| CBOW      |  0.458|  0.497|  0.556|  0.637|  0.630|
| CMOW      |  0.432|  0.334|  0.403|  0.471|  0.529|
| Hybrid    |  0.472|  0.476|  0.530|  0.621|  0.613|

### Train CBOW, CMOW, and CBOW-CMOW hybrid model

To train a 784-dimensional CBOW model, run the following:

```
python train_cbow.py --w2m_type cbow --batch_size=1024 --outputdir=<path_to_save_model> --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1000 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 784 --temp_path <some_directory_for_temp_files> --dataset_path=<path_to_parsed_UMBC_dataset> --num_workers 2 --output_file <path_to_output.csv> --num_docs 134442680 --stop_criterion train_loss
```

For CMOW:

```
python train_cbow.py --w2m_type cmow --batch_size=1024 --outputdir=<path_to_save_model> --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1000 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 784 --temp_path <some_directory_for_temp_files> --dataset_path=<path_to_parsed_UMBC_dataset> --num_workers 2 --output_file <path_to_output.csv> --num_docs 134442680 --stop_criterion train_loss --initialization identity
```

And the CBOW-CMOW Hybrid:

```
python train_cbow.py --w2m_type hybrid --batch_size=1024 --outputdir=<path_to_save_model> --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1000 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 400 --temp_path <some_directory_for_temp_files> --dataset_path=<path_to_parsed_UMBC_dataset> --num_workers 2 --output_file <path_to_output.csv> --num_docs 134442680 --stop_criterion train_loss --initialization identity
```

### Evaluate components of hybrid model

In the paper, we have shown that the jointly training of the individual CBOW/CMOW components emphasizes their individual strengths.
To assess the performance of the CBOW component, restrict the final embedding representation to include only the first
half of the representations from the HybridEncoder (--included_features 0 400 in a 800-dimensional Hybrid encoder), or restrict it to the second half (--included features 400 800) to evaluate the CMOW component.
E.g, for evaluating the CMOW component, run:

```
python evaluate_word2mat.py --encoders <path_to_hybrid.encoder_file> --word_vocab <path_to_.vocab_file> --included_features 400 800 --outputdir <temp_path_to_save_encoder> --outputmodelname hybrid_constituent --downstream_eval full
```

Here, 'encoder' and 'word_vocab' is saved in 'outputdir' after training the models. By

## Files

- `train_cbow.py` Main training executable. Type python train_cbow.py --help to get overview of training parameters.
- `cbow.py` Contains the data preparation code as well as the neural architecture for CBOW except the encoder.
- `word2mat.py` The code for word2mat encoder.
- `wrap_evaluation.py` Wrapper script for SentEval to automatically evaluate encoder after training.
- `evaluate_word2mat.py` Script for evaluating sub-components of hybrid encoder with SentEval.
- `mutils.py` Helpers for saving the results, hyperparameter optimization and stuff.

## Reference

Please cite our ICLR paper [[1]](https://openreview.net/forum?id=H1MgjoR9tQ) to reference our work or code.

### CBOW Is Not All You Need: Combining CBOW with the Compositional Matrix Space Model (ICLR 2019)

[1] Mai, F., Galke, L & Scherp, A., [*CBOW Is Not All You Need: Combining CBOW with the Compositional Matrix Space Model*](https://openreview.net/forum?id=H1MgjoR9tQ)

```
@inproceedings{mai2018cbow,
title={{CBOW} Is Not All You Need: Combining {CBOW} with the Compositional Matrix Space Model},
author={Florian Mai and Lukas Galke and Ansgar Scherp},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=H1MgjoR9tQ},
}
```
