import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer

from official_eval import official_f1

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds, train_idx):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(train_idx+1 + idx, relation_labels[pred]))


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels, eval_dir,eval_script):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels, eval_dir,eval_script)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, eval_dir,eval_script,average="macro"):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
        "f1": official_f1(eval_dir, eval_script),
    }

def my_calculate_score(df, labels,**kwargs):
    from sklearn.metrics import precision_recall_fscore_support as f1_score
    from sklearn.metrics import confusion_matrix as confusion_matrix
    import warnings
    import sklearn.exceptions
    import re
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    import numpy as np

    # labels = sorted(df['Y'].unique())

    if "ignore_direction" in kwargs and kwargs.get("ignore_direction"):
        Y=df.Y.str.replace(r'\(.*?\)', r'', regex=True)
        pred_Y = df["^Y"].str.replace(r'\(.*?\)', r'', regex=True)
        labels=list(dict.fromkeys([re.sub("\(.*?\)", "", _) for _ in labels]))

    else:
        Y = df["Y"]
        pred_Y = df[kwargs.get("threshold") if "threshold" in kwargs else "^Y"]

    precision, recall, fscore, support = f1_score(Y, pred_Y, labels=labels)

    confM = confusion_matrix(Y, pred_Y)

    accuracy=confM.diagonal()/confM.sum(axis=1)

    return dict(labels=labels,
                precision=precision.tolist(),
                recall=recall.tolist(),
                fscore=fscore.tolist(),
                support=support.tolist(),
                accuracy=accuracy.tolist(),
                confusion_matrix=confM.tolist()) #ndarray cannot be saved as jsonfile

