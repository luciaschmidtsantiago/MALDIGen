import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.conditional_utils import get_condition, impute_missing_labels, compute_attr_prediction_loss

############### GAN ###############

class MLPDecoder1D_Generator(nn.Module):
    def __init__(self, latent_dim, num_layers, output_dim, cond_dim=0, use_bn=False):
        """ Flexible MLP generator (decoder) with optional Batch Normalization.
        Args:
            latent_dim (int): Dimension of the latent noise vector.
            num_layers (int): Number of hidden layers.
            output_dim (int): Dimension of the output spectrum.
            cond_dim (int): Dimension of conditional vector (0 for unconditional).
            use_bn (bool): Whether to include BatchNorm1d layers.
        """
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.in_dim = latent_dim + cond_dim
        self.use_bn = use_bn

        layers = []
        in_dim = self.in_dim

        for i in range(num_layers):
            out_dim = 2 ** (i + 2) * latent_dim  # 4×, 8×, 16×...
            layers.append(nn.Linear(in_dim, out_dim))
            if self.use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = out_dim

        # Output layer: maps to final spectrum dimension
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())  # normalize output between [0,1]

        self.net = nn.Sequential(*layers)

    def forward(self, z, cond=None):
        """ Forward pass through the generator.
        Args:
            z (torch.Tensor): Latent noise tensor [batch, latent_dim].
            cond (torch.Tensor, optional): Conditional vector [batch, cond_dim].
        Returns: torch.Tensor: Generated output [batch, output_dim].
        """
        if cond is not None:
            z = torch.cat([z, cond], dim=1)
        return self.net(z)

class CNNDecoder1D_Generator(nn.Module):
    """
    Simple 1D CNN Generator for spectra synthesis.
    Architecture:
      1. Linear layer expands latent vector to feature map.
      2. Three upsampling + Conv1d blocks increase length and reduce channels.
      3. Conditional vector (if provided) is concatenated after convolutions.
      4. Final MLP head projects to output spectrum.
    """
    def __init__(
        self,
        latent_dim,
        output_dim,
        n_layers=3,
        cond_dim=0,
        base_channels=128,
        use_dropout=True,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.base_channels = base_channels
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        # Compute initial feature map size
        init_length = 16  # You may want to make this dynamic
        # Fix: first conv layer should output 64 channels, not base_channels
        # Set up channel progression: [128, 64, 32, 16, ...]
        out_channels = [max(1, base_channels // (2 ** (i + 1))) for i in range(n_layers)]
        in_channels = [base_channels] + out_channels[:-1]

        # 1. Linear: latent_dim → base_channels * init_length
        self.fc = nn.Linear(latent_dim, base_channels * init_length)
        self.init_length = init_length

        # 2. Build upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        for i in range(n_layers):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(in_channels[i], out_channels[i], kernel_size=5, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_prob) if use_dropout else nn.Identity()
            )
            self.upsample_blocks.append(block)

        # 3. Compute flattened size for final MLP
        with torch.no_grad():
            dummy = torch.zeros(1, latent_dim)
            h = self.fc(dummy)
            x = h.view(1, base_channels, init_length)
            for block in self.upsample_blocks:
                x = block(x)
            flattened_dim = x.view(1, -1).shape[1]

        # 4. Final MLP head
        self.final_fc = nn.Sequential(
            nn.Linear(flattened_dim + cond_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z, cond=None):
        """
        Forward pass for generalizable CNN-based generator.
        Args:
            z (Tensor): Latent noise [batch, latent_dim]
            cond (Tensor, optional): Conditional vector [batch, cond_dim]
        Returns:
            out (Tensor): Generated spectrum [batch, output_dim]
        """
        # 1. Latent vector → feature map
        h = self.fc(z)
        x = h.view(z.size(0), self.base_channels, self.init_length)

        # 2. CNN upsampling path
        for block in self.upsample_blocks:
            x = block(x)

        # 3. Flatten and concatenate condition
        x = x.view(z.size(0), -1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # 4. Final MLP head
        out = self.final_fc(x)
        return out

# Discriminator: D(x, y)
class Discriminator(nn.Module):
    def __init__(self, image_dim, cond_dim=0, use_bn=False, use_dropout=True, dropout_prob=0.1):
        super().__init__()
        input_dim = image_dim + cond_dim
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 1)
        if use_bn:
            self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(1024), nn.BatchNorm1d(512), nn.BatchNorm1d(256)
        else:
            self.bn1 = self.bn2 = self.bn3 = None
        if use_dropout:
            self.dp1, self.dp2, self.dp3 = nn.Dropout(dropout_prob), nn.Dropout(dropout_prob), nn.Dropout(dropout_prob)
        else:
            self.dp1 = self.dp2 = self.dp3 = None

    def forward(self, x, cond=None):
        if cond is not None and cond.numel() > 0:
            x = torch.cat([x, cond], dim=1)
        for fc, bn, dp in zip([self.fc1, self.fc2, self.fc3], [self.bn1, self.bn2, self.bn3], [self.dp1, self.dp2, self.dp3]):
            x = fc(x)
            if bn: x = bn(x)
            x = F.leaky_relu(x, 0.2)
            if dp: x = dp(x)
        # Pass through Sigmoid
        out = torch.sigmoid(self.fc_out(x))
        return out
    
################ GAN ###############
class GAN(nn.Module):
    """
    Simple (unconditional) GAN wrapper class for training and evaluation.

    Args:
        generator (nn.Module): generator network (G)
        discriminator (nn.Module): discriminator network (D)
    """
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward_G(self, z):
        """
        Forward through the generator.
        Args:
            z (Tensor): latent vector [batch, latent_dim]
        Returns:
            Tensor: generated sample [batch, output_dim]
        """
        return self.generator(z)

    def forward_D(self, x):
        """
        Forward through the discriminator.
        Args:
            x (Tensor): real or generated data [batch, input_dim]
        Returns:
            Tensor: discriminator output (probability of being real)
        """
        return self.discriminator(x)


################ Conditional GAN ###############
class ConditionalGAN(nn.Module):
    """
    Conditional GAN wrapper that manages conditioning embeddings and 
    passes cond vectors to G and D automatically.
    """
    def __init__(self, generator, discriminator, 
                 y_species_dim, y_embed_dim, y_amr_dim=0, embedding=True):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.embedding = embedding

        # Embedding layers
        self.y_embed_species = nn.Embedding(y_species_dim, y_embed_dim) if embedding else None
        self.y_embed_amr = nn.Embedding(y_amr_dim, y_embed_dim) if (embedding and y_amr_dim > 0) else None
        self.y_amr_dim = y_amr_dim

    def get_cond(self, y_species, y_amr=None):
        return get_condition(
            y_species, y_amr, 
            self.y_embed_species, self.y_embed_amr,
            embedding=self.embedding
        )

    def forward_G(self, z, y_species=None, y_amr=None):
        cond = self.get_cond(y_species, y_amr) if y_species is not None else None
        return self.generator(z, cond)

    def forward_D(self, x, y_species=None, y_amr=None):
        cond = self.get_cond(y_species, y_amr) if y_species is not None else None
        return self.discriminator(x, cond)
    
def generate_spectra_per_label_cgan(model, label_correspondence, n_samples, latent_dim, device=None):
    """
    Generate n_samples spectra for each label using a ConditionalGAN.
    Args:
        model: Trained ConditionalGAN (must be in eval mode).
        label_correspondence: dict mapping label indices to label names (or vice versa).
        n_samples: Number of spectra to generate per label.
        latent_dim: Dimension of the latent noise vector.
        device: torch.device (optional, will use model's device if None).
    Returns:
        dict: {label_name: tensor of generated spectra}
    """
    # model.eval()
    device = device or next(model.parameters()).device
    results = {}
    for idx, label_name in label_correspondence.items():
        y_species = torch.full((n_samples,), idx, dtype=torch.long, device=device)
        z = torch.randn(n_samples, latent_dim, device=device)
        with torch.no_grad():
            generated = model.forward_G(z, y_species)
        results[label_name] = generated.detach().cpu()
    return results



################ Semi-supervised Conditional GAN ###############

class SemisupervisedConditionalGAN(ConditionalGAN):
    def __init__(self, generator, discriminator, attr_predictor, latent_dim=128, lambda_g=1.0, lambda_d=1.0, alpha=1.0, missing_strategy='soft'):
        # Inicializar ConditionalGAN base
        super().__init__(generator, discriminator, latent_dim, lambda_g, lambda_d)
        
        # Inferir parámetros automáticamente desde los componentes
        y_dim = generator.y_embed.num_embeddings
        y_embed_dim = generator.y_embed.embedding_dim
        self.label2_dim = generator.label2_dim
        
        # Añadir componentes específicos para semisupervised
        self.attr_predictor = attr_predictor
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.alpha = alpha  # peso para attr_prediction loss
        
        # missing_strategy controla cómo manejar missing values para la GAN:
        # - 'soft': Usar predicciones como probabilidades soft → embedding ponderado
        # - 'hard': Umbralizar predicciones a 0/1 → embedding tradicional  
        # - 'ignore': Missing → 0, solo usar observados → más conservador
        assert missing_strategy in ['soft', 'hard', 'ignore'], f"missing_strategy debe ser 'soft', 'hard' o 'ignore', got {missing_strategy}"
        self.missing_strategy = missing_strategy
    
    def forward(self, x_real, y, label2):
        # 1. Imputar missing values según la estrategia configurada
        y_input = impute_missing_labels(x_real, y, label2, self.attr_predictor, self.y_embed, self.label2_dim, self.missing_strategy)
        
        # 2. La GAN usa directamente las etiquetas procesadas
        # get_condition automáticamente detecta si son soft o hard labels
        gan_loss = super().forward(x_real, y_input, label2)
        
        # 3. Pérdida de predicción siempre sobre atributos observados reales
        attr_pred_loss = compute_attr_prediction_loss(x_real, y, label2, self.attr_predictor, self.y_embed, self.label2_dim)
        
        total_loss = gan_loss + self.alpha * attr_pred_loss
        return total_loss
    
    def sample(self, y, label2, batch_size=64, fill_missing='neutral'):
        """
        Generar muestras reutilizando ConditionalGAN.sample después de imputar
        fill_missing: 'neutral' (missing → 0.5) o 'predictor' (usar attr_predictor)
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
        
        # Reutilizar el método sample del ConditionalGAN padre
        return super().sample(y_filled, label2, batch_size)