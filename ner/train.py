"""
Script to train model.
"""
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer, BertForTokenClassification, AdamW,
    get_linear_schedule_with_warmup)
from seqeval.metrics import accuracy_score
from bedrock_client.bedrock.api import BedrockApi

from utils import load_data, convert_examples_to_features

DATA_DIR = "gs://bedrock-sample/ner_data/"
PRETRAINED_MODEL_NAME = "bert-base-multilingual-cased"
MAX_SEQUENCE_LENGTH = 200
LABEL_LIST = ["O", "PERSON", "ORGANIZATION", "LOCATION", "[CLS]", "[SEP]", "X"]
REV_LABEL_MAP = {i: label for i, label in enumerate(LABEL_LIST, 1)}
NUM_LABELS = len(LABEL_LIST) + 1
EPOCHS = 4
BATCH_SIZE = 16
LR = 2e-5
WARMUP = 0.1
LOGGING_STEPS = 20
ACCUMULATION_STEPS = 1
FINETUNED_MODEL_PATH = "/artefact/finetuned_bert.bin"


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
    data_loader = DataLoader(
        data_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return data_loader


def evaluate_model(model, val_loader, device):
    """Evaluate model."""
    val_loss = 0
    nb_val_steps = 0
    y_true = []
    y_pred = []
    for batch in val_loader:
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


def train_model(model, train_loader, val_loader, device):
    """Train model."""
    max_grad_norm = 1.0
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]

    num_total_steps = int(EPOCHS * len(train_loader) / ACCUMULATION_STEPS)
    num_warmup_steps = WARMUP * num_total_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_total_steps)  # PyTorch scheduler

    best_loss = np.inf
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    for epoch in range(EPOCHS):
        # TRAIN loop
        t0 = time.time()
        model.train()
        
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_id = batch
            # forward pass
            outputs = model(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=label_id)
            
            loss = outputs[0]
            if ACCUMULATION_STEPS > 1:
                loss = loss / ACCUMULATION_STEPS

            # backward pass
            loss.backward()

            # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            # update parameters
            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()  # same as optimizer.zero_grad()
                global_step += 1

            if global_step % LOGGING_STEPS == 0:
                loss_scalar = (tr_loss - logging_loss) / LOGGING_STEPS
                logging_loss = tr_loss
                print(f"Epoch {epoch + 1}: global step = {global_step}  train loss = {loss_scalar:.4f}")

        model.eval()
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}: elapsed time = {time.time() - t0:.0f}s"
              f"  val loss = {val_loss:.4f}  val accuracy = {val_acc:.4f}")
        
        if val_loss < best_loss:
            # Save model artefact
            print(f"Epoch {epoch + 1}: val loss improved from {best_loss:.5f} to {val_loss:.5f}, "
                  f"saving model to {FINETUNED_MODEL_PATH}\n")
            best_loss = val_loss
            best_acc = val_acc
            torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
        else:
            print(f"Epoch {epoch + 1}: val loss did not improve from {best_loss:.5f}\n")
            
    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Val_loss", best_loss)
    bedrock.log_metric("Val_accuracy", best_acc)


def train():
    """Train pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.device_count() > 0:
        print(f"  Found GPU at: {torch.cuda.get_device_name(0)}")

    print("\nLoad data")
    train_examples = load_data(DATA_DIR + "ner_train_data.txt")
    print("  Training data size =", len(train_examples))

    val_examples = load_data(DATA_DIR + "ner_val_data.txt")
    print("  Validation data size =", len(val_examples))

    print("\nTokenize")
    bert_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=False)
    train_features = convert_examples_to_features(
        train_examples, LABEL_LIST, MAX_SEQUENCE_LENGTH, bert_tokenizer)
    val_features = convert_examples_to_features(
        val_examples, LABEL_LIST, MAX_SEQUENCE_LENGTH, bert_tokenizer)

    print("\nTrain model")
    train_loader = get_dataloader(train_features, BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_features, BATCH_SIZE)
    model = BertForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)
    train_model(model, train_loader, val_loader, device)


if __name__ == "__main__":
    train()
