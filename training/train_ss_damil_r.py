import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import DualRoleBagDataset, collate_dual_role_bags, load_interviews_with_roles
from models.ss_damil_r import SSDamilRClassifier
from training.train_damil_r import set_seed, precompute_dual_role_embeddings
from utils.metrics import compute_metrics, find_best_threshold


class CorrectFocalLoss(nn.Module):
    """Focal Loss with proper alpha class balancing."""
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        at = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        f_loss = at * (1 - pt) ** self.gamma * bce_loss
        return f_loss.mean() if self.reduction == 'mean' else f_loss


class WeightedSymptomLoss(nn.Module):
    """v5: Auxiliary Task (Binary Classification) with Weighted BCE."""
    
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, symptom_logits, symptom_targets, has_symptoms):
        """
        Args:
            symptom_logits: (B, 8)
            symptom_targets: (B, 8) containing values 0-3
            has_symptoms: (B,) mask
        """
        # Threshold targets to binary (0 vs 1+)
        binary_targets = (symptom_targets > 0).float()
        
        # BCE with logits
        # F.binary_cross_entropy_with_logits handles pos_weight correctly for binary tasks
        loss = F.binary_cross_entropy_with_logits(
            symptom_logits, binary_targets, 
            pos_weight=self.pos_weight, reduction='none'
        ) # (B, 8)
        
        # Mean loss per interview, then mask
        loss_per_interview = loss.mean(dim=1)
        weighted_loss = (loss_per_interview * has_symptoms).sum()
        mask_sum = has_symptoms.sum()
        
        return weighted_loss / (mask_sum + 1e-8) if mask_sum > 0 else torch.tensor(0.0, device=symptom_logits.device)


class MultiTaskLoss(nn.Module):
    """v8: Main (Focal) + Symptom (Weighted BCE)."""
    
    def __init__(
        self, 
        main_loss_fn, 
        symptom_loss_fn,
        aux_weight=0.5
    ):
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.symptom_loss_fn = symptom_loss_fn
        self.aux_weight = aux_weight

    def forward(
        self, 
        logits, targets, 
        symptom_logits, symptom_targets, has_symptoms
    ):
        # 1. Main Task (Depression binary)
        main_loss = self.main_loss_fn(logits, targets)

        # 2. Auxiliary Task (Binary Symptom Classification)
        avg_aux_loss = self.symptom_loss_fn(symptom_logits, symptom_targets, has_symptoms)

        total_loss = main_loss + self.aux_weight * avg_aux_loss
        
        return total_loss, main_loss, avg_aux_loss, torch.tensor(0.0, device=logits.device)


def train_epoch(
    model, loader, criterion, optimizer, device,
    max_grad_norm=1.0, noise_std=0.0
):
    model.train()
    total_loss, total_m_loss, total_a_loss, total_d_loss = 0, 0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        optimizer.zero_grad()
        
        target = batch["labels"].to(device)
        sym_target = batch["symptoms"].to(device)
        has_sym = batch["has_symptoms"].to(device)

        output = model.forward_batch(
            batch["patient_bags"].to(device),
            batch["interviewer_bags"].to(device),
            batch["patient_sizes"],
            batch["interviewer_sizes"],
            noise_std=noise_std,
        )

        loss, ml, al, dl = criterion(
            output["logits"], target, 
            output["symptom_logits"], sym_target, has_sym
        )

        if not torch.isfinite(loss):
            logging.warning(f"  [WARN] Non-finite loss detected. Skipping batch.")
            continue

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        params_with_nan = [p for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if params_with_nan:
            logging.warning(f"  [WARN] NaNs in gradients! Skipping optimizer step.")
            optimizer.zero_grad()
            continue

        optimizer.step()

        total_loss += loss.item()
        total_m_loss += ml.item()
        total_a_loss += al.item()
        total_d_loss += dl.item()
        
        probs = torch.sigmoid(output["logits"]).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(target.cpu().numpy())

    return (
        total_loss/len(loader), total_m_loss/len(loader), 
        total_a_loss/len(loader), total_d_loss/len(loader),
        compute_metrics(np.array(all_labels), (np.array(all_probs) >= 0.5).astype(int), np.array(all_probs))
    )


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, total_ml = 0, 0
    all_probs, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            output = model.forward_batch(
                batch["patient_bags"].to(device),
                batch["interviewer_bags"].to(device),
                batch["patient_sizes"],
                batch["interviewer_sizes"]
            )
            target = batch["labels"].to(device)
            loss, ml, _, _ = criterion(
                output["logits"], target,
                output["symptom_logits"], batch["symptoms"].to(device),
                batch["has_symptoms"].to(device)
            )
            total_loss += loss.item()
            total_ml += ml.item()
            
            probs = torch.sigmoid(output["logits"]).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(target.cpu().numpy())
            all_ids.extend(batch["interview_ids"])

    y_true, y_prob = np.array(all_labels), np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    v_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    v_ml = total_ml / len(loader) if len(loader) > 0 else 0.0
    if not np.isfinite(v_loss):
        v_loss = 9.999

    return (
        v_loss, v_ml, 
        compute_metrics(y_true, y_pred, y_prob), 
        {"probability": all_probs, "true_label": all_labels}
    )


def compute_symptom_weights(interviews):
    """Computes binary class weights (0 vs 1+) across all 8 symptoms."""
    counts = np.zeros(2) # 0: negative, 1: positive
    for iv in interviews:
        if iv.get("has_symptoms", False):
            for s_val in iv["symptoms"]:
                idx = 1 if s_val > 0 else 0
                counts[idx] += 1
    
    total = counts.sum()
    if total == 0 or counts[1] == 0:
        return torch.tensor([1.0])
    
    # pos_weight for BCE is neg_counts / pos_counts
    pos_weight = counts[0] / max(1, counts[1])
    return torch.tensor([pos_weight], dtype=torch.float32)

