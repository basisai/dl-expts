"""
Script to train model.
"""
import logging
import re
import time

from bedrock_client.bedrock.api import BedrockApi
import torch
import torch.utils.data
from pytorch_transformers import (
    BertTokenizer, BertForTokenClassification, AdamW, WarmupLinearSchedule)
from seqeval.metrics import accuracy_score

MODEL_NAME = "bert-base-multilingual-cased"
MAX_SEQUENCE_LENGTH = 200
LABEL_LIST = ["O", "PERSON", "ORGANIZATION", "LOCATION", "[CLS]", "[SEP]", "X"]
REV_LABEL_MAP = {i: label for i, label in enumerate(LABEL_LIST, 1)}
NUM_LABELS = len(LABEL_LIST) + 1
EPOCHS = 4
BATCH_SIZE = 16
LR = 2e-5
WARMUP = 0.1
ACCUMULATION_STEPS = 1
OUTPUT_MODEL_PATH = "bert_pytorch.bin"


def whitespace_punctuation(s):
    """Add whitespace before punctuation."""
    s = re.sub(r"([.,!?()])", r" \1 ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def word_tagging(raw_line):
    """"Tag each word for given line."""
    raw_line = re.sub(r"\t\d+\n", "", raw_line)
    line_split0 = re.split(r'<ENAMEX TYPE="(.*?)">(.*?)</ENAMEX>', raw_line)

    raw_tags = []
    line_split1 = []
    flag = 0
    for x in line_split0:
        if x not in ["PERSON", "ORGANIZATION", "LOCATION"]:
            if flag == 0:
                raw_tags.append("O")
            line_split1.append(x)
            flag = 0
        else:
            flag = 1
            raw_tags.append(x)

    line_split = []
    tags = []
    for x, t in zip(line_split1, raw_tags):
        y = whitespace_punctuation(x).split()
        line_split.extend(y)
        tags.extend([t] * len(y))
    return line_split, tags


class InputExample:
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def load_data(file_path):
    """Load data."""
    data_examples = []
    with open(file_path, "r") as file:
        for i, raw_line in enumerate(file):
            line_split, tags = word_tagging(raw_line)
            data_examples.append(InputExample(i, " ".join(line_split), label=tags))
    return data_examples


class InputFeatures:
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# pylint: disable=too-many-locals
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for ex_index, example in enumerate(examples):
        textlist = example.text_a.split(" ")
        labellist = example.label
        if labellist is None:
            labellist = ["O"] * len(textlist)
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_id = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_id.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_id.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_id.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_id.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_id) == max_seq_length

        if ex_index < 3:
            print("*** Example ***")
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def evaluate_model(model, val_dataloader, device):
    """Evaluate model."""
    val_loss = 0
    nb_val_steps = 0
    y_true = []
    y_pred = []
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_id = batch

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, labels=label_id)
            loss, logits = outputs[:2]

        val_loss += loss.item()
        nb_val_steps += 1

        indices = torch.argmax(logits, dim=2).detach().cpu().numpy()
        input_mask = input_mask.to("cpu").numpy()
        label_id = label_id.to("cpu").numpy()

        for i, mask in enumerate(input_mask):
            tmp_true = []
            tmp_pred = []
            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if REV_LABEL_MAP[label_id[i][j]] != "X":
                        tmp_true.append(REV_LABEL_MAP[label_id[i][j]])
                        tmp_pred.append(REV_LABEL_MAP[indices[i][j]])
                else:
                    tmp_true.pop()
                    tmp_pred.pop()
                    break
            y_true.append(tmp_true)
            y_pred.append(tmp_pred)

    val_loss /= nb_val_steps
    val_acc = accuracy_score(y_true, y_pred)
    return val_loss, val_acc, y_true, y_pred


def get_dataloader(data_features, batch_size, shuffle=False, drop_last=False):
    """Output dataloader."""
    data_input_ids = torch.tensor(
        [f.input_ids for f in data_features], dtype=torch.long)
    data_input_mask = torch.tensor(
        [f.input_mask for f in data_features], dtype=torch.long)
    data_segment_ids = torch.tensor(
        [f.segment_ids for f in data_features], dtype=torch.long)
    data_label_id = torch.tensor(
        [f.label_id for f in data_features], dtype=torch.long)
    data_dataset = torch.utils.data.TensorDataset(
        data_input_ids, data_input_mask, data_segment_ids, data_label_id)
    data_dataloader = torch.utils.data.DataLoader(
        data_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return data_dataloader


def train_model(train_features, val_features, device):
    """Train model."""
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS)
    model.zero_grad()
    model.to(device)

    max_grad_norm = 1.0
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    num_total_steps = int(
        EPOCHS * len(train_features) / BATCH_SIZE / ACCUMULATION_STEPS)
    num_warmup_steps = WARMUP * num_total_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, correct_bias=False)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    train_dataloader = get_dataloader(train_features, BATCH_SIZE, shuffle=True)
    val_dataloader = get_dataloader(val_features, BATCH_SIZE)

    for epoch in range(EPOCHS):
        # TRAIN loop
        start = time.time()
        model.train()

        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_id = batch
            # forward pass
            outputs = model(input_ids, segment_ids, input_mask, label_id)
            loss = outputs[0]
            if ACCUMULATION_STEPS > 1:
                loss = loss / ACCUMULATION_STEPS

            # backward pass
            loss.backward()

            # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            nb_tr_steps += 1

            # update parameters
            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print("Epoch {}: Train loss = {}".format(epoch, tr_loss / nb_tr_steps))
        print("\tTime taken = {:.2f} mins".format((time.time() - start) / 60))

        print("Evaluate")
        model.eval()
        val_loss, val_acc, _, _ = evaluate_model(model, val_dataloader, device)
        print("Val loss = {}".format(val_loss))
        print("Val accuracy = {}".format(val_acc))

        # Log metrics
        bedrock = BedrockApi(logging.getLogger(__name__))
        bedrock.log_metric("Eval_accuracy", val_acc)
        bedrock.log_metric("Eval_loss", val_loss)

    # Save model artefact
    torch.save(model.state_dict(), "/artefact/" + OUTPUT_MODEL_PATH)


def main():
    """Train pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.device_count() > 0:
        print("\tFound GPU at: {}".format(torch.cuda.get_device_name(0)))

    print("\nLoad data")
    train_examples = load_data("data/ner_train_data.txt")
    print("Training data size =", len(train_examples))

    val_examples = load_data("data/ner_val_data.txt")
    print("Validation data size =", len(val_examples))

    print("\nTokenize")
    bert_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    print("Train examples")
    train_features = convert_examples_to_features(
        train_examples, LABEL_LIST, MAX_SEQUENCE_LENGTH, bert_tokenizer)
    print("Val examples")
    val_features = convert_examples_to_features(
        val_examples, LABEL_LIST, MAX_SEQUENCE_LENGTH, bert_tokenizer)

    print("\nTrain")
    train_model(train_features, val_features, device)


if __name__ == "__main__":
    main()
