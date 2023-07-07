# semantic_extents

Repository for the "Explaining Relation Classification Models with Semantic
Extents" research paper

## Environment

The following setup aims to detect problems with cuda (Nvidia GPU) access as
early as possible.

However, all necessary dependencies are in the `env.yml` file. feel free to
install the Conda environment. In most cases this should work as well.

### Automatic setup

We also provide a bash script which automates the environment setup.
Just run

```bash
source ./dev-setup.sh
```

and you should be ready to go. The script creates and/or updates the
environment `reex`,  
activates the environment, and checks for cuda availability. If the command was
successful you can jump direcly to
Model Training.

### Step by step version

Install a Python 3.9 environment with AllenNLP as pip requirement. This should
install
PyTorch with GPU access, if you have an Nvidia GPU and all drivers installed.
You can check by executing `nvidia-smi`.

```bash
conda create -n reex python=3.9 conda-build

conda activate reex

pip install allennlp==2.10.0
```

Afterward, lets check if cuda is available in PyTorch. Open a Python console by
executing `python`.
Now enter following lines and compare outputs.

```python
import torch

torch.cuda.is_available()
```

Result should be `True`.

Now Install remaining dependencies.

```bash
pip install -r requirements.txt

conda develop .

python -m spacy download en_core_web_sm
```

## Model Training

To train a model, comparable to the RoBERTa model from the paper, follow these
steps:

### 1. Specify paths in configuration file

Change the `train_data_path`, `validation_data_path` and `test_data_path`
in `configuration/base_config.jsonnet`.
Sets paths to preprocessed Versions of the ACE 05 dataset. After following the
instruction in the [preprocessing](https://github.com/MSLars/ace_preprocessing) repo, you
find the `train.jsonl`, `dev.jsonl` and `test.jsonl` files in
the `preprocessed_relation_classification_data`
directory.

Your configuration could look like:

```
{
  train_data_path: "<path to cloned repo>/ace_preprocessing/preprocessed_relation_classification_data/train.jsonl",
  validation_data_path: "<path to cloned repo>/ace_preprocessing/preprocessed_relation_classification_data/dev.jsonl",
  test_data_path: "<path to cloned repo>/ace_preprocessing/preprocessed_relation_classification_data/test.jsonl"
}
```

Specify whether to use a GPU for training (-1 for no GPU or 0...n for a specific
GPU.
For example 0 if you have only one available GPU). Default is GPU training.

```
trainer: {
    validation_metric: "+f1-macro-overall",
    num_epochs: 50,
    patience: 5,
    cuda_device: 0, # -1 or GPU Index, default is CPU training
    optimizer: {
        type: "huggingface_adamw",
        lr: learn_rate,
    },
}
```

If your GPU runs out of memory, you can change the batch size in the
configuration.

```
    data_loader: {
        batch_sampler:{
            batch_size: 32, # Change this to a smaller value if necessary
            type: 'bucket',
        },
    },
```

You can also specify whether to train base or large version by changing the
comment in the first
two lines of the config file. By default we train the base version.

```
# local bert_model = "roberta-large";
local bert_model = "roberta-base"; # here we train the base version
```

Change the ` -s <OUT PATH>` parameter to your desired location.

```
allennlp train configuration/base_config.jsonnet -s <OUT PATH> --include-package reex
```

Especially on CPU training is time consuming, we recomment GPU training.

## Load pretrained model

Finetuned based version: https://fh-aachen.sciebo.de/s/H8rnAeOBp8M6Zmq

Finetuned Large version: https://fh-aachen.sciebo.de/s/rCreVe4yw7LjjOb

You can see the models metrics in the directory `model_meta`.

To reproduce further results, such as the model extent etc. move
the `model_base.tar.gz`
file in the `pretrained_models` directory.

## Model Evaluation

To Evaluate models on the complete test set, insert link to model file, for
example `model_base.tar.gz`
and link to text dataset, preprocessed with ace_preprocessingh repository.
Execute following command.

Test data path could look like
this: `../ace_preprocessing/preprocessed_relation_classification_data/test.jsonl `.

Replace `base_evaluation.josn` with your desired output file path.

```
allennlp evaluate <PATH_TO .tar.gz> <PATH_TO_TEST_DATA> --include-package reex --output-file base_evaluation.json
```

Result should be

```
  ...
  "f1-macro-overall": 0.6786807400208933,
  "f1-micro-overall": 0.8326967358589172,
}
```

which are the results for the base model in the paper.

## Computing Model Semantic Extent Evaluations

To compute the semantic extents for a model, specify

- Path to model archive, for example, `pretrained_models/model_base.tar.gz`
- Path to test data, for example, `../ace_preprocessing/preprocessed_relation_classification_data/test.jsonl`
- Path to output result. select file in `explanaitions` directory to reproduce further results conveniently.

and execute the following command

```
python explain/explanaition.py <PATH_TO .tar.gz> <PATH_TO_TEST_DATA> <RESULT_JSONL_PATH>
```

You might want to reduce the number of evaluation samples with the `--subset_size` parameter.
computation might take up to ~1 Minute per sample.

Here is a short documentation of the resulting datastructure:

#### Explanaition

This is the top level datastructure and contains all information we compute during evaluation for each classification sample.

```
{
    "sample": original relation classification sample,
    "full_confidence": analysis of prediction on full sample,
    "extended_confidence": analysis on relation extent from ace 05 dataset (see original annotation guidelines for further details),
    "reduced_confidence": analysis on input reduction sample. Input reduction from adversarial attacker from AllenNLP Interpret module,
    "indices_in_reduction": Indices of tokens in the original sample that are in the result of the input reduction.
    "saliency_interpretations": Different saliency scores from AllenNLP Interpret module. Simple, smooth and integrated gradient.
    "hir_samples": Extension candidates for semantic extent. For each of these we make a model prediction and check if the model makes the same prediction on reduced input and on complete sample.
    "candidate_area_confidences": As described in paper, model predictions only on the tokens of current extent candidate,
    "candidate_outside_shuffled_confidences": The tokens in current extent candidate are keept, but all other tokens are shuffled outside of the arguments.
    "candidate_complete_shuffled_confidences": The tokens of extent are kept in their original position. All other tokens are shuffeld accross the non fixed positions.    
}
```

#### Confidences

For each model decision we compute a so-called confidence object

```
{
    "prediction": Predicted Label,
    "prediction_idx": Index in vocab of predicted Label,
    "confidence": Probability of highest predicted prob.,
    "probabilities": Predicted prob for each label,
    "gold_label": Gold Label,
    "gold_label_idx": Index in vocab of gold label,
    "model_tokens": Tokens (based on spacy tokenization as in ace_preprocessing repo),
    "pieces2model_tokens": mapping from wordpieces to original tokens. [-1, 0, 1 1] means that first wordpiece does not occur in original sample (CLS token) 3rd and 4th wordpiece both belong to 2nd original token ...,
}
```

#### Extent Candidates

The extent candidates as they are described and reffered to in the paper are called hir_samples
in this JSON structure. This is due to later renaming.

```
{   
    "tokens": tokens in current extent candidate,
    "relation": start and end index for arguments and the relation label,
    "description": type of extent (ONLY_ARGUMENTS, if it contains only the arguments, ARGS_SUBTREE, if it contains syntactical subtree tokens noun phrases etc. of arguments further docs in paper),
    "count_description", how many tokens with this calss follow? (current index/ total tokens of type),
    "token_idx": index of tokens in original sample
}
```