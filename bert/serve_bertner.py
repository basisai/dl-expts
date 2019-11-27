"""
Script for serving.
"""
import json
import os
import socketserver
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler

import torch
from transformers import BertTokenizer, BertForTokenClassification
from jieba import cut

from train_bertner import convert_examples_to_features, whitespace_punctuation, InputExample

SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))

MODEL_NAME = "bert-base-multilingual-cased"
MAX_SEQUENCE_LENGTH = 200
LABEL_LIST = ["O", "PERSON", "ORGANIZATION", "LOCATION", "[CLS]", "[SEP]", "X"]
REV_LABEL_MAP = {i: label for i, label in enumerate(LABEL_LIST, 1)}
NUM_LABELS = len(LABEL_LIST) + 1

BERT_TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

MODEL = BertForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS)
MODEL.load_state_dict(torch.load("/artefact/bert_pytorch.bin", map_location=torch.device("cpu")))
for param in MODEL.parameters():
    param.requires_grad = False
MODEL.eval()


def split_text(text, lang):
    """Split text."""
    if lang in ["zh-tw", "zh-cn"]:
        return [el for el in cut(text, cut_all=False) if el != " "]
    return whitespace_punctuation(text).split()


# pylint: disable=too-many-locals
def predict(text, lang=None, model=MODEL, tokenizer=BERT_TOKENIZER):
    """Predict."""
    text_split = split_text(text, lang)
    sam_features = convert_examples_to_features(
        [InputExample(0, " ".join(text_split), label=["O"] * len(text_split))],
        LABEL_LIST, MAX_SEQUENCE_LENGTH, tokenizer)

    sam_input_ids = torch.tensor([f.input_ids for f in sam_features], dtype=torch.long)
    sam_input_mask = torch.tensor([f.input_mask for f in sam_features], dtype=torch.long)
    sam_segment_ids = torch.tensor([f.segment_ids for f in sam_features], dtype=torch.long)

    with torch.no_grad():
        logits = model(
            sam_input_ids,
            token_type_ids=sam_segment_ids,
            attention_mask=sam_input_mask,
            labels=None)[0]

    indices = torch.argmax(logits, dim=2).detach().numpy()[0]
    mask = sam_features[0].input_mask
    label_id = sam_features[0].label_id

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
class Handler(SimpleHTTPRequestHandler):
    """Handler for http requests"""

    def do_POST(self):
        """Returns a re-ranked list of items given features."""
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        self.send_response(HTTPStatus.OK)
        self.end_headers()

        post_data = json.loads(post_data.decode("utf-8"))
        text_split, y_pred = predict(
            post_data["text"],
            post_data.get("lang"),
        )
        result = {
            "text_split": text_split,
            "tags": y_pred
        }
        self.wfile.write(bytes(json.dumps(result).encode("utf-8")))


def main():
    """Starts the Http server"""
    print("Starting server at {}".format(SERVER_PORT))
    httpd = socketserver.TCPServer(("", SERVER_PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
