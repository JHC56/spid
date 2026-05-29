"""SPID — training: Round 1 + Round 2 (hard-negative mining)."""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support

from config import (
    SEED, MODEL_NAME, MAX_LEN, DEVICE,
    CLASS_WEIGHTS_LIST, LABEL_SMOOTHING,
    R1_EPOCHS, R1_LR, R1_BATCH, R1_GRAD_ACCUM, R1_WARMUP,
    R2_EPOCHS, R2_LR, R2_BATCH, R2_GRAD_ACCUM, R2_WARMUP,
    WEIGHT_DECAY,
)
from utils import build_dataset, load_ood_data

#  Weighted Trainer 

CLASS_WEIGHTS = torch.tensor(CLASS_WEIGHTS_LIST).to(DEVICE)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits, labels,
            weight=CLASS_WEIGHTS,
            label_smoothing=LABEL_SMOOTHING,
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": p, "recall": r, "f1": f1}


#  Main training loop

def train():
    # data 
    train_list, test_list, hnm_cand_atk, hnm_cand_nrm, all_attacks, all_normals = (
        build_dataset()
    )
    ood_data, ood_labels = load_ood_data()

    #  tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LEN, padding=False
        )

    train_tok = Dataset.from_list(train_list).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    test_tok = Dataset.from_list(test_list).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    ood_tok = Dataset.from_list(ood_data).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"train: {len(train_tok)}, test: {len(test_tok)}, ood: {len(ood_tok)}")

    # model 
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,} (full fine-tuning)")
    print(f"loss: CE (weight {CLASS_WEIGHTS_LIST}, LS {LABEL_SMOOTHING})")

    # Round 1 
    print("\n=== Round 1: standard training ===")
    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./spid_r1",
            num_train_epochs=R1_EPOCHS,
            per_device_train_batch_size=R1_BATCH,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=R1_GRAD_ACCUM,
            learning_rate=R1_LR,
            warmup_ratio=R1_WARMUP,
            weight_decay=WEIGHT_DECAY,
            logging_steps=20,
            eval_strategy="no",
            save_strategy="no",
            fp16=True,
            report_to="none",
            seed=SEED,
        ),
        train_dataset=train_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("Round 1 complete.")

    # Round 2: Hard Negative Mining 
    print("\n=== Round 2: Hard Negative Mining ===")

    if len(hnm_cand_atk) > 0 and len(hnm_cand_nrm) > 0:
        hnm_data = (
            [{"text": t, "label": 1} for t in hnm_cand_atk]
            + [{"text": t, "label": 0} for t in hnm_cand_nrm]
        )
        hnm_tok = Dataset.from_list(hnm_data).map(
            tokenize, batched=True, remove_columns=["text"]
        )

        hnm_pred = trainer.predict(
            hnm_tok, ignore_keys=["hidden_states", "attentions"]
        )
        hnm_probs = torch.softmax(
            torch.tensor(hnm_pred.predictions), dim=-1
        ).numpy()
        hnm_labels_arr = np.array([d["label"] for d in hnm_data])
        hnm_preds = (hnm_probs[:, 1] > 0.5).astype(int)

        hard_examples = [
            hnm_data[i] for i in range(len(hnm_data))
            if hnm_preds[i] != hnm_labels_arr[i]
        ]
        borderline = [
            hnm_data[i] for i in range(len(hnm_data))
            if 0.3 < hnm_probs[i, 1] < 0.7
            and hnm_preds[i] == hnm_labels_arr[i]
        ]

        print(f"  hard examples (model wrong):   {len(hard_examples)}")
        print(f"  borderline examples (0.3-0.7): {len(borderline)}")

        hnm_train_data = train_list + hard_examples + borderline
        random.shuffle(hnm_train_data)
        hnm_train_tok = Dataset.from_list(hnm_train_data).map(
            tokenize, batched=True, remove_columns=["text"]
        )
        print(f"  Round 2 training set: {len(hnm_train_tok)} "
              f"(orig {len(train_list)} + HNM {len(hard_examples)+len(borderline)})")

        trainer2 = WeightedTrainer(
            model=model,
            args=TrainingArguments(
                output_dir="./spid_r2",
                num_train_epochs=R2_EPOCHS,
                per_device_train_batch_size=R2_BATCH,
                per_device_eval_batch_size=32,
                gradient_accumulation_steps=R2_GRAD_ACCUM,
                learning_rate=R2_LR,
                warmup_ratio=R2_WARMUP,
                weight_decay=WEIGHT_DECAY,
                logging_steps=20,
                eval_strategy="no",
                save_strategy="no",
                fp16=True,
                report_to="none",
                seed=SEED,
            ),
            train_dataset=hnm_train_tok,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer2.train()
        trainer = trainer2
        print("\n=== Training complete (Round 1 + HNM Round 2) ===")
    else:
        print("  (no HNM candidates available, skipping Round 2)")
        print("\n=== Training complete (Round 1 only) ===")

    # Save model 
    save_dir = "./spid-deberta-base"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # save temperature placeholder (evaluate.py will overwrite with real value)
    with open(os.path.join(save_dir, "spid_config.json"), "w") as f:
        json.dump({
            "temperature": 1.0,
            "threshold": 0.5,
            "base_model": MODEL_NAME,
        }, f)

    total_mb = sum(
        os.path.getsize(os.path.join(save_dir, f))
        for f in os.listdir(save_dir)
    ) / 1e6
    print(f"\nsaved: {save_dir} ({total_mb:.0f} MB)")

    return model, tokenizer, trainer, train_list, test_list, ood_data, ood_labels, ood_tok, test_tok


if __name__ == "__main__":
    train()
