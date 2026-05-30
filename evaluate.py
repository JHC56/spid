"""SPID — evaluation: calibration, OOD eval (SPID only, no ensemble)."""

import gc
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report,
)

from config import DEVICE, MAX_LEN, TARGET_PRECISION, MIN_RECALL
from utils import split_sentence


#  Temperature calibration

def calibrate_temperature(trainer, test_tok, test_list):
    """Find optimal temperature on held-out test set."""
    test_pred = trainer.predict(
        test_tok, ignore_keys=["hidden_states", "attentions"]
    )
    test_logits = torch.tensor(test_pred.predictions)
    test_labels_t = torch.tensor([x["label"] for x in test_list])

    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.1, 5.0, 0.1):
        scaled = test_logits / t
        nll = F.cross_entropy(scaled, test_labels_t).item()
        if nll < best_nll:
            best_nll = nll
            best_t = t

    print(f"optimal temperature: {best_t:.1f}")

    test_probs_calibrated = torch.softmax(
        test_logits / best_t, dim=-1
    ).numpy()

    return best_t, test_probs_calibrated


#  In-distribution threshold sweep

def indist_sweep(test_probs_calibrated, test_list):
    """Threshold sweep on in-distribution test set."""
    test_labels = np.array([x["label"] for x in test_list])

    print("=== threshold sweep (in-distribution, calibrated) ===")
    print(f"{'thr':>6} {'prec':>8} {'recall':>8} {'f1':>8}")
    for thr in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]:
        p = (test_probs_calibrated[:, 1] >= thr).astype(int)
        pr = precision_score(test_labels, p, zero_division=0)
        rc = recall_score(test_labels, p, zero_division=0)
        f1 = f1_score(test_labels, p, zero_division=0)
        print(f"{thr:>6.2f} {pr:>8.4f} {rc:>8.4f} {f1:>8.4f}")


#  OOD evaluation (SPID only)

def evaluate_ood(trainer, ood_tok, ood_data, ood_labels, temperature):
    """OOD sweep for SPID alone."""

    ood_pred = trainer.predict(
        ood_tok, ignore_keys=["hidden_states", "attentions"]
    )
    ood_logits = torch.tensor(ood_pred.predictions)
    ood_probs_calibrated = torch.softmax(
        ood_logits / temperature, dim=-1
    ).numpy()

    print(f"=== OOD threshold sweep (SPID only, T={temperature:.1f}) ===")
    print(f"{'thr':>6} {'prec':>8} {'recall':>8} {'f1':>8} {'block':>8}")
    
    threshold = None
    for thr in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        p = (ood_probs_calibrated[:, 1] >= thr).astype(int)
        pr = precision_score(ood_labels, p, zero_division=0)
        rc = recall_score(ood_labels, p, zero_division=0)
        f1 = f1_score(ood_labels, p, zero_division=0)
        bl = p.mean()
        meets = pr >= TARGET_PRECISION and rc >= MIN_RECALL
        if meets and threshold is None:
            threshold = thr
        note = "<<< selected" if meets else ""
        print(f"{thr:>6.2f} {pr:>8.4f} {rc:>8.4f} {f1:>8.4f} {bl:>8.1%} {note}")
    
    if threshold is None:
        # Fallback: best F1
        best_f1 = 0
        for thr in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            p = (ood_probs_calibrated[:, 1] >= thr).astype(int)
            f1 = f1_score(ood_labels, p, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                threshold = thr
        print(f"\nfallback (best F1): threshold={threshold}")

    # Final report 
    print("\n" + "=" * 60)
    print(f"OOD FINAL — SPID (threshold={threshold}, T={temperature:.1f})")
    print("=" * 60)
    spid_preds_final = (ood_probs_calibrated[:, 1] >= threshold).astype(int)
    print(classification_report(
        ood_labels, spid_preds_final,
        target_names=["safe", "unsafe"], digits=4,
    ))
    print(f"block rate: {spid_preds_final.mean():.4f}")

    # Update saved config 
    cfg_path = os.path.join("./spid-deberta-base", "spid_config.json")
    os.makedirs("./spid-deberta-base", exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump({
            "temperature": float(temperature),
            "threshold": float(threshold),
            "base_model": "microsoft/deberta-v3-base",
        }, f)
    print(f"\nsaved config: T={temperature}, thr={threshold}")

    return threshold, temperature, ood_probs_calibrated, spid_preds_final
