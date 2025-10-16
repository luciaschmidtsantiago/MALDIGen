import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_condition(y_species, y_amr=None, y_embed_layer_species=None, y_embed_layer_amr=None, embedding=False, n_classes=6):
    """
    Get conditioning vector for conditional VAE.

    Args:
        y_species (Tensor): Long tensor of shape [batch] with categorical species labels (0 to num_species-1).
        y_amr (Tensor or None): Optional multi-label or soft-label tensor [batch, num_amr_labels].
        y_embed_layer_species (nn.Embedding): Embedding layer for species.
        y_embed_layer_amr (nn.Embedding): Embedding layer for AMR labels (multi-label).
        embedding (bool): If True, use embedding representations. If False, return one-hot.

    Returns:
        cond (Tensor): Conditioning vector [batch, cond_dim]
    """

    if embedding:
        # Embed species (always categorical)
        y_species_emb = y_embed_layer_species(y_species)  # [batch, embed_dim_species]

        if y_amr is not None and y_amr.shape[1] > 0:
            all_amr_embeds = y_embed_layer_amr.weight  # [num_amr, embed_dim]
            batch_embeds = []

            for i in range(y_amr.size(0)):
                y_sample = y_amr[i]
                if torch.any((y_sample > 0) & (y_sample < 1)):
                    weighted_embeds = y_sample.unsqueeze(1) * all_amr_embeds
                    prob_sum = y_sample.sum()
                    if prob_sum > 0:
                        amr_emb = weighted_embeds.sum(dim=0) / prob_sum
                    else:
                        amr_emb = torch.zeros(y_embed_layer_amr.embedding_dim, device=y_sample.device)
                else:
                    idxs = (y_sample == 1).nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:
                        amr_emb = torch.zeros(y_embed_layer_amr.embedding_dim, device=y_sample.device)
                    else:
                        amr_emb = y_embed_layer_amr(idxs).mean(dim=0)
                batch_embeds.append(amr_emb)

            y_amr_emb = torch.stack(batch_embeds, dim=0)  # [batch, embed_dim_amr]
            cond = torch.cat([y_species_emb, y_amr_emb], dim=1)
        else:
            cond = y_species_emb

    # One-hot encoding without embedding
    else:
        # y_species is categorical multi-class
        y_species_onehot = F.one_hot(y_species, num_classes=n_classes).float()
        if y_amr is not None and y_amr.shape[1] > 0:
            # y_amr is expected to be multi-label binary or soft labels
            cond = torch.cat([y_species_onehot, y_amr], dim=1)
        else:
            cond = y_species_onehot

    return cond

#### OLD VERSION OF GET CONDITION ####
def get_condition_old(y, label2, y_embed_layer, label2_dim):
    """
    Método común para generar condiciones a partir de etiquetas multilabel y categóricas.
    Ahora soporta tanto etiquetas binarias (0/1) como probabilidades soft [0,1].
    
    Args:
        y: tensor [batch, num_labels] con etiquetas multilabel 
           - Valores 0/1: etiquetas binarias (comportamiento original)
           - Valores [0,1]: probabilidades soft (nuevo comportamiento)
        label2: tensor [batch] con etiquetas categóricas
        y_embed_layer: nn.Embedding layer para las etiquetas multilabel
        label2_dim: número de categorías para label2
    
    Returns:
        cond: tensor [batch, y_embed_dim + label2_dim] con la condición concatenada
    """
    # Procesar etiquetas multilabel (con soporte para soft labels)
    batch_embeds = []
    
    # Obtener todos los embeddings de una vez para eficiencia
    all_label_embeds = y_embed_layer.weight  # [num_labels, embed_dim]
    
    for i in range(y.size(0)):
        y_sample = y[i]  # [num_labels]
        
        # Si hay probabilidades soft (valores entre 0 y 1), usar ponderación
        # Si son binarias (solo 0s y 1s), comportamiento original
        if torch.any((y_sample > 0) & (y_sample < 1)):
            # SOFT LABELS: Ponderar embeddings por probabilidades
            # y_sample[j] * all_label_embeds[j] para cada label j
            weighted_embeds = y_sample.unsqueeze(1) * all_label_embeds  # [num_labels, embed_dim]
            
            # Suma ponderada normalizada por la suma de probabilidades
            prob_sum = y_sample.sum()
            if prob_sum > 0:
                y_emb_sample = weighted_embeds.sum(dim=0) / prob_sum  # [embed_dim]
            else:
                y_emb_sample = torch.zeros(y_embed_layer.embedding_dim, device=y.device)
        else:
            # HARD LABELS: Comportamiento original (solo 0s y 1s)
            idxs = (y_sample == 1).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                y_emb_sample = torch.zeros(y_embed_layer.embedding_dim, device=y.device)
            else:
                embeds = y_embed_layer(idxs)
                y_emb_sample = embeds.mean(dim=0)
        
        batch_embeds.append(y_emb_sample)
    
    y_emb = torch.stack(batch_embeds, dim=0)
    
    # Procesar etiquetas categóricas
    label2_onehot = F.one_hot(label2, num_classes=label2_dim).float()
    
    # Concatenar ambas representaciones
    cond = torch.cat([y_emb, label2_onehot], dim=1)
    return cond

def impute_missing_labels(x, y, label2, attr_predictor, y_embed, label2_dim, missing_strategy):
    """
    Función común para imputar missing values (-1) usando attr_predictor.
    Retorna etiquetas según la estrategia configurada.
    
    Args:
        x: tensor [batch, input_dim] datos de entrada
        y: tensor [batch, num_labels] etiquetas multilabel con missing (-1)
        label2: tensor [batch] etiquetas categóricas
        attr_predictor: red neuronal para predecir atributos
        y_embed: nn.Embedding layer para atributos
        label2_dim: número de categorías para label2
        missing_strategy: 'soft', 'hard' o 'ignore'
    
    Returns:
        y_filled: tensor [batch, num_labels] etiquetas imputadas
    """
    y_filled = y.clone().float()
    
    for i in range(y.size(1)):
        missing_idx = (y[:, i] == -1).nonzero(as_tuple=True)[0]
        if len(missing_idx) > 0:
            if missing_strategy == 'ignore':
                # Solo cambiar -1 → 0 (ignorar missing)
                y_filled[missing_idx, i] = 0.0
            else:
                # Usar attr_predictor para imputar
                x_missing = x[missing_idx]
                y_idx = torch.full((len(missing_idx),), i, dtype=torch.long, device=x.device)
                y_emb = y_embed(y_idx)
                label2_missing = label2[missing_idx]
                label2_onehot = F.one_hot(label2_missing, num_classes=label2_dim).float()
                attr_input = torch.cat([x_missing, y_emb, label2_onehot], dim=1)
                pred = attr_predictor(attr_input).squeeze(-1)
                
                if missing_strategy == 'soft':
                    # Usar probabilidades directamente (soft labels)
                    y_filled[missing_idx, i] = pred
                elif missing_strategy == 'hard':
                    # Umbralizar a 0/1 (hard labels)
                    y_filled[missing_idx, i] = (pred > 0.5).float()
    
    return y_filled

def compute_attr_prediction_loss(x, y, label2, attr_predictor, y_embed, label2_dim):
    """
    Función común para computar BCE loss para atributos observados (no missing).
    
    Args:
        x: tensor [batch, input_dim] datos de entrada
        y: tensor [batch, num_labels] etiquetas multilabel (con missing como -1)
        label2: tensor [batch] etiquetas categóricas
        attr_predictor: red neuronal para predecir atributos
        y_embed: nn.Embedding layer para atributos
        label2_dim: número de categorías para label2
    
    Returns:
        attr_pred_loss: pérdida de predicción de atributos
    """
    attr_pred_loss = 0.0
    
    for i in range(y.size(1)):
        observed_idx = (y[:, i] != -1).nonzero(as_tuple=True)[0]
        if len(observed_idx) > 0:
            x_obs = x[observed_idx]
            y_idx = torch.full((len(observed_idx),), i, dtype=torch.long, device=x.device)
            y_emb = y_embed(y_idx)
            label2_obs = label2[observed_idx]
            label2_onehot = F.one_hot(label2_obs, num_classes=label2_dim).float()
            attr_input = torch.cat([x_obs, y_emb, label2_onehot], dim=1)
            pred = attr_predictor(attr_input)
            target = y[observed_idx, i].float()
            attr_pred_loss += F.binary_cross_entropy(pred.squeeze(-1), target)
    
    return attr_pred_loss
