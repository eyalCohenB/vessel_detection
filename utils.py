# utils.py
import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()

def compute_loss(presence_logit, column_logits, y_presence, y_column, lambda_col=1.0):
    loss_presence = bce_loss(presence_logit, y_presence)

    positive_mask = (y_presence == 1)

    if positive_mask.any():
        loss_column = ce_loss(
            column_logits[positive_mask],
            y_column[positive_mask]
        )
    else:
        loss_column = torch.tensor(0.0, device=presence_logit.device)

    total_loss = loss_presence + lambda_col * loss_column

    return total_loss, loss_presence, loss_column