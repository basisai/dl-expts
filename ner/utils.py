"""
Script containing commonly used functions.
"""
import re

import pandas as pd


def load_data(file_path):
    """Load data."""
    df = pd.read_csv(file_path, sep="\t", names=["text", "id"])

    data_examples = []
    for i, raw_line in enumerate(df["text"].values):
        line_split, tags = word_tagging(raw_line)
        data_examples.append(InputExample(i, " ".join(line_split), label=tags))
    return data_examples


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


class InputFeatures:
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# pylint: disable=too-many-locals
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 print_examples=False):
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

        if print_examples and ex_index < 3:
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
