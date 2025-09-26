import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cVAE import ConditionalVAE

class SemisupervisedConditionalVAE(ConditionalVAE):
    def __init__(self, encoder, decoder, prior, attr_predictor, alpha=1.0, beta=1.0, missing_strategy='soft'):
        # Inicializar ConditionalVAE base
        super().__init__(encoder, decoder, prior, beta)
        
        # Inferir parámetros automáticamente desde los componentes
        y_dim = encoder.y_embed.num_embeddings
        y_embed_dim = encoder.y_embed.embedding_dim
        self.label2_dim = encoder.label2_dim
        
        # Añadir componentes específicos para semisupervised
        self.attr_predictor = attr_predictor
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.alpha = alpha         # peso para attr_prediction loss
        
        # missing_strategy controla cómo manejar missing values para el VAE:
        # - 'soft': Usar predicciones como probabilidades soft → embedding ponderado
        # - 'hard': Umbralizar predicciones a 0/1 → embedding tradicional  
        # - 'ignore': Missing → 0, solo usar observados → más conservador
        assert missing_strategy in ['soft', 'hard', 'ignore'], f"missing_strategy debe ser 'soft', 'hard' o 'ignore', got {missing_strategy}"
        self.missing_strategy = missing_strategy
    
    def _impute_missing_labels(self, x, y, label2):
        """
        Imputar missing values (-1) usando attr_predictor.
        Retorna etiquetas según la estrategia configurada.
        """
        y_filled = y.clone().float()
        
        for i in range(y.size(1)):
            missing_idx = (y[:, i] == -1).nonzero(as_tuple=True)[0]
            if len(missing_idx) > 0:
                if self.missing_strategy == 'ignore':
                    # Solo cambiar -1 → 0 (ignorar missing)
                    y_filled[missing_idx, i] = 0.0
                else:
                    # Usar attr_predictor para imputar
                    x_missing = x[missing_idx]
                    y_idx = torch.full((len(missing_idx),), i, dtype=torch.long, device=x.device)
                    y_emb = self.y_embed(y_idx)
                    label2_missing = label2[missing_idx]
                    label2_onehot = F.one_hot(label2_missing, num_classes=self.label2_dim).float()
                    attr_input = torch.cat([x_missing, y_emb, label2_onehot], dim=1)
                    pred = self.attr_predictor(attr_input).squeeze(-1)
                    
                    if self.missing_strategy == 'soft':
                        # Usar probabilidades directamente (soft labels)
                        y_filled[missing_idx, i] = pred
                    elif self.missing_strategy == 'hard':
                        # Umbralizar a 0/1 (hard labels)
                        y_filled[missing_idx, i] = (pred > 0.5).float()
        
        return y_filled
    
    def _compute_attr_prediction_loss(self, x, y, label2):
        """Computar BCE loss para atributos observados (no missing)"""
        attr_pred_loss = 0.0
        
        for i in range(y.size(1)):
            observed_idx = (y[:, i] != -1).nonzero(as_tuple=True)[0]
            if len(observed_idx) > 0:
                x_obs = x[observed_idx]
                y_idx = torch.full((len(observed_idx),), i, dtype=torch.long, device=x.device)
                y_emb = self.y_embed(y_idx)
                label2_obs = label2[observed_idx]
                label2_onehot = F.one_hot(label2_obs, num_classes=self.label2_dim).float()
                attr_input = torch.cat([x_obs, y_emb, label2_onehot], dim=1)
                pred = self.attr_predictor(attr_input)
                target = y[observed_idx, i].float()
                attr_pred_loss += F.binary_cross_entropy(pred.squeeze(-1), target)
        
        return attr_pred_loss
    
    def forward(self, x, y, label2):
        # 1. Imputar missing values según la estrategia configurada
        y_input = self._impute_missing_labels(x, y, label2)
        
        # 2. El VAE usa directamente las etiquetas procesadas
        # get_condition automáticamente detecta si son soft o hard labels
        vae_loss = super().forward(x, y_input, label2)
        
        # 3. Pérdida de predicción siempre sobre atributos observados reales
        attr_pred_loss = self._compute_attr_prediction_loss(x, y, label2)
        
        total_loss = vae_loss + self.alpha * attr_pred_loss
        return total_loss
    
    def sample(self, y, label2, batch_size=64, fill_missing='neutral'):
        """
        Generar muestras reutilizando ConditionalVAE.sample después de imputar
        fill_missing: 'neutral' (missing → 0.5) o 'predictor' (usar attr_predictor)
        
        Nota: En generación, use_filled_labels no aplica porque siempre necesitamos
        etiquetas completas para generar. La decisión es cómo llenar los missing.
        """
        # Imputar missing values para generación
        y_filled = y.clone().float()
        
        for i in range(y.size(1)):
            missing_idx = (y[:, i] == -1).nonzero(as_tuple=True)[0]
            if len(missing_idx) > 0:
                if fill_missing == 'predictor':
                    # Usar predictor (necesita input x dummy para generación)
                    x_dummy = torch.zeros((len(missing_idx), 9701), device=y.device)  # TODO: hacer dinámico
                    y_idx = torch.full((len(missing_idx),), i, dtype=torch.long, device=y.device)
                    y_emb = self.y_embed(y_idx)
                    label2_missing = label2[missing_idx]
                    label2_onehot = F.one_hot(label2_missing, num_classes=self.label2_dim).float()
                    attr_input = torch.cat([x_dummy, y_emb, label2_onehot], dim=1)
                    pred = self.attr_predictor(attr_input)
                    y_filled[missing_idx, i] = pred.squeeze(-1)
                else:
                    y_filled[missing_idx, i] = 0.5  # neutral/incierto
        
        # Reutilizar el método sample del ConditionalVAE padre
        return super().sample(y_filled, label2, batch_size)
