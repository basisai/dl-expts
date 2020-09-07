"""
Script for serving.
"""
import os

import torch
from flask import Flask, request
from transformers import BertTokenizer, BertForTokenClassification
from jieba import cut

from utils import convert_examples_to_features, whitespace_punctuation, InputExample

MODEL_DIR = "/artefact/"
if os.path.exists("models/"):
    MODEL_DIR = "models/"
FINETUNED_MODEL_PATH = MODEL_DIR + "finetuned_bert.bin"

PRETRAINED_MODEL_NAME = "bert-base-multilingual-cased"
MAX_SEQUENCE_LENGTH = 200
LABEL_LIST = ["O", "PERSON", "ORGANIZATION", "LOCATION", "[CLS]", "[SEP]", "X"]
REV_LABEL_MAP = {i: label for i, label in enumerate(LABEL_LIST, 1)}
NUM_LABELS = len(LABEL_LIST) + 1

BERT_TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=False)

MODEL = BertForTokenClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
MODEL.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=torch.device("cpu")))
MODEL.eval()


def split_text(text, lang="en"):
    """Split text."""
    if lang in ["zh-tw", "zh-cn"]:
        return [el for el in cut(text, cut_all=False) if el != " "]
    return whitespace_punctuation(text).split()


# pylint: disable=too-many-locals
def predict(request_json):
    """Predict."""
    text = request_json["text"]
    lang = request_json.get("lang")
        
    text_split = split_text(text, lang)
    features = convert_examples_to_features(
        [InputExample(0, " ".join(text_split), label=["O"] * len(text_split))],
        LABEL_LIST, MAX_SEQUENCE_LENGTH, BERT_TOKENIZER)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    with torch.no_grad():
        logits = MODEL(input_ids, segment_ids, input_mask, labels=None)[0]

    indices = torch.argmax(logits, dim=2).detach().numpy()[0]
    mask = features[0].input_mask
    label_id = features[0].label_id

    tags = []
    for j, m in enumerate(mask):
        if j == 0:
            continue
        if m:
            if REV_LABEL_MAP[label_id[j]] != "X":
                tags.append(REV_LABEL_MAP[indices[j]])
        else:
            tags.pop()
            break
    return text_split, tags


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    text_split, tags = predict(request.json)
    return {"text_split": text_split, "tags": tags}


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
