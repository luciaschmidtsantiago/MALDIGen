import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.conditional_utils import get_condition, impute_missing_labels, compute_attr_prediction_loss

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        # handle zero-dim input (no context) gracefully
        if self.input_dim == 0:
            # return a zeros tensor of shape (batch, emb_dim)
            batch = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
            emb_dim = self.model[-1].out_features if hasattr(self.model[-1], 'out_features') else self.model[-1].in_features
            return x.new_zeros((batch, emb_dim))
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)
    
class ContextUnet1D(nn.Module):
    """
    Original asymmetric 1D U-Net for diffusion, now configurable.
    Reproduces exactly your architecture for n_blocks=2 and n_feat=64.
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_feat: int = 64,
        n_cfeat: int = 6,
        length: int = 6000,
        n_blocks: int = 2,
        norm_groups: int = 8,
        kernel_size: int = 4,
        logger=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.length = length
        self.n_blocks = n_blocks
        self.logger = logger

        # --- Initial 1D conv ---
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, n_feat),
            nn.ReLU()
        )

        # --- Down path (fixed two-level pattern for now) ---
        self.down_blocks = nn.ModuleList()
        in_ch = n_feat
        down_channels = [n_feat]  # keep track of channels at each scale
        for i in range(n_blocks):
            out_ch = in_ch if i < n_blocks - 1 else 2 * in_ch  # last down doubles channels
            self.down_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1),
                nn.GroupNorm(norm_groups, out_ch),
                nn.ReLU()
            ))
            down_channels.append(out_ch)
            in_ch = out_ch

        # --- Bottleneck ---
        self.to_vec = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.GELU()
        )

        # --- Embeddings (same placement/order as original) ---
        # self.timeembed1    = EmbedFC(1, 2 * n_feat)
        # self.timeembed2    = EmbedFC(1, n_feat)
        # self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        # self.contextembed2 = EmbedFC(n_cfeat, n_feat)
        self.timeembeds = nn.ModuleList()
        self.contextembeds = nn.ModuleList()
        for i in range(self.n_blocks):
            # progressively smaller embedding sizes as channels shrink
            out_dim = 2 * self.n_feat if i == 0 else self.n_feat
            self.timeembeds.append(EmbedFC(1, out_dim))
            self.contextembeds.append(EmbedFC(self.n_cfeat, out_dim))

        # --- Up path ---
        # Big upsample from vector to L/4 (for 2 downs) or L/(2**n_blocks)
        self.up0 = nn.Sequential(
            nn.ConvTranspose1d(2 * n_feat, 2 * n_feat,
                               kernel_size=length // (2 ** n_blocks),
                               stride=length // (2 ** n_blocks)),
            nn.GroupNorm(norm_groups, 2 * n_feat),
            nn.ReLU()
        )

        # Dynamically build upsampling path (mirror of down path)
        self.up_blocks = nn.ModuleList()
        in_ch = 2 * n_feat  # from up0
        for i in range(n_blocks):
            skip_ch = down_channels[-(i + 1)]
            out_ch = n_feat if i < n_blocks - 1 else n_feat  # constant decoder width
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch + skip_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(norm_groups, out_ch),
                nn.ReLU()
            ))
            in_ch = out_ch

        # # Standard stride=2 deconvs (mirror of down path)
        # self.up_blocks = nn.ModuleList()
        # self.up_blocks.append(nn.Sequential(
        #     nn.ConvTranspose1d(4 * n_feat, n_feat, kernel_size=4, stride=2, padding=1),
        #     nn.GroupNorm(norm_groups, n_feat),
        #     nn.ReLU()
        # ))
        # self.up_blocks.append(nn.Sequential(
        #     nn.ConvTranspose1d(2 * n_feat, n_feat, kernel_size=4, stride=2, padding=1),
        #     nn.GroupNorm(norm_groups, n_feat),
        #     nn.ReLU()
        # ))

        # --- Output layer ---
        self.out = nn.Sequential(
            nn.Conv1d(2 * n_feat, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, n_feat),
            nn.ReLU(),
            nn.Conv1d(n_feat, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, c=None):
        # --- Encode ---
        x = self.init_conv(x)
        downs = [x]
        for down in self.down_blocks:
            downs.append(down(downs[-1]))

        # --- Bottleneck ---
        hiddenvec = self.to_vec(downs[-1])

        # --- Default context ---
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat, device=x.device, dtype=x.dtype)

        # # --- Embeddings (all right after GELU, before decoding) ---
        # cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1)
        # temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1)
        # cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1)
        # temb2 = self.timeembed2(t).view(-1, self.n_feat, 1)

        # --- Decoder ---
        up = self.up0(hiddenvec)

        # Loop through each upsampling block with its own embeddings
        for i, up_block in enumerate(self.up_blocks):
            skip = downs[-(i + 1)]
            
            # Compute embeddings for this scale
            cemb = self.contextembeds[i](c).view(c.size(0), -1, 1)
            temb = self.timeembeds[i](t).view(t.size(0), -1, 1)
            
            up = cemb * up + temb

            # Match lengths before concat
            if up.shape[-1] != skip.shape[-1]:
                min_len = min(up.shape[-1], skip.shape[-1])
                up = up[..., :min_len]
                skip = skip[..., :min_len]
            
            # Concatenate and upsample
            up = up_block(torch.cat([up, skip], dim=1))

        # Final output
        out = self.out(torch.cat([up, downs[0]], dim=1))
        return out

    # def forward(self, x, t, c=None):
    #     # Encode
    #     x = self.init_conv(x)
    #     downs = [x]
    #     for down in self.down_blocks:
    #         downs.append(down(downs[-1]))

    #     # Bottleneck
    #     hiddenvec = self.to_vec(downs[-1])

    #     # Default context
    #     if c is None:
    #         c = torch.zeros(x.shape[0], self.n_cfeat, device=x.device, dtype=x.dtype)

    #     # Embeddings
    #     cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1)
    #     temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1)
    #     cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1)
    #     temb2 = self.timeembed2(t).view(-1, self.n_feat, 1)

    #     # Decode
    #     up1 = self.up0(hiddenvec)
    #     up2 = self.up_blocks[0](torch.cat([cemb1 * up1 + temb1, downs[-1]], dim=1))
    #     up3 = self.up_blocks[1](torch.cat([cemb2 * up2 + temb2, downs[-2]], dim=1))
    #     out = self.out(torch.cat([up3, downs[0]], dim=1))
    #     return out

def generate_spectra_per_label_ddpm(model, label_correspondence, n_samples, timesteps, a_t, b_t, ab_t, logger, device):
    """
    Generate n_samples per label using the trained diffusion model.
    """
    model.eval()
    results = {}
    num_classes = len(label_correspondence)

    for label_id, label_name in label_correspondence.items():
        logger.info(f"Generating diffusion samples for label: {label_name}") if logger else None

        # --- Create correct one-hot context for that label ---
        c = torch.zeros(n_samples, num_classes, device=device)
        c[:, label_id] = 1.0

        # --- Start from Gaussian noise ---
        L = model.length
        x = torch.randn(n_samples, model.in_channels, L, device=device)

        # --- Diffusion sampling ---
        with torch.no_grad():
            for t_inv in range(timesteps, 0, -1):
                t = torch.full((n_samples,), t_inv, device=device, dtype=torch.long)
                t_norm = (t.float() / float(timesteps)).view(-1, 1)
                eps = model(x, t_norm, c)
                ab = ab_t[t].view(n_samples, 1, 1)
                a = a_t[t].view(n_samples, 1, 1)
                b = b_t[t].view(n_samples, 1, 1)
                x = (x - (b / (1 - ab).sqrt()) * eps) / a.sqrt()
                if t_inv > 1:
                    x += b.sqrt() * torch.randn_like(x)

        results[label_name] = x.detach().cpu()

    return results

# OUTDATED
class ConditionalDiffusion(nn.Module):
    """
    Conditional Diffusion Model que sigue la misma estructura que VAE/GAN
    Mantiene API consistente: forward() retorna loss escalar
    """
    def __init__(self, denoiser, y_dim, y_embed_dim, label2_dim, timesteps=500, beta1=1e-4, beta2=0.02):
        super().__init__()
        self.denoiser = denoiser  # ContextUnet o similar
        self.y_dim = y_dim
        self.y_embed_dim = y_embed_dim
        self.label2_dim = label2_dim
        self.timesteps = timesteps
        
        # Embedding para etiquetas multilabel (mismo que VAE/GAN)
        self.y_embed = nn.Embedding(2, y_embed_dim)  # Para valores 0/1
        
        # Diffusion schedule (DDPM)
        device = next(denoiser.parameters()).device
        b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()
        ab_t[0] = 1
        
        # Register as buffers (no gradients, moved with model)
        self.register_buffer('b_t', b_t)
        self.register_buffer('a_t', a_t)
        self.register_buffer('ab_t', ab_t)
    
    def get_condition(self, y, label2):
        """Reutiliza la función común para generar condiciones"""
        return get_condition(y, label2, self.y_embed, self.label2_dim)
    
    def perturb_input(self, x, t, noise):
        """Aplica ruido a los datos en el timestep t (funciona para imágenes y datos vectorizados)"""
        # x: [batch, ...]  t: [batch]  noise: same shape as x
        
        # Adaptar indexing según las dimensiones de x
        if len(x.shape) == 4:
            # Formato imagen: [batch, channels, height, width]
            alpha_bar_t = self.ab_t.sqrt()[t, None, None, None]  
            one_minus_alpha_bar_t = (1 - self.ab_t[t, None, None, None]).sqrt()
        elif len(x.shape) == 3:
            # Formato 3D: [batch, channels, length] 
            alpha_bar_t = self.ab_t.sqrt()[t, None, None]
            one_minus_alpha_bar_t = (1 - self.ab_t[t, None, None]).sqrt()
        elif len(x.shape) == 2:
            # Formato vectorizado: [batch, features]
            alpha_bar_t = self.ab_t.sqrt()[t, None]
            one_minus_alpha_bar_t = (1 - self.ab_t[t, None]).sqrt()
        else:
            raise ValueError(f"Formato de entrada no soportado: {x.shape}. Se esperan 2D (vectorizado), 3D o 4D (imágenes)")
        
        scaled_x = alpha_bar_t * x
        scaled_noise = one_minus_alpha_bar_t * noise
        return scaled_x + scaled_noise
    
    def forward(self, x, y, label2):
        """
        Forward pass para entrenamiento (compatible con funciones genéricas)
        Args:
            x: imágenes [batch, D] o [batch, C, H, W]
            y: etiquetas multilabel [batch, y_dim] 
            label2: etiquetas categóricas [batch]
        Returns:
            loss: tensor escalar (MSE loss del diffusion)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Generar condición (mismo formato que VAE/GAN)
        cond = self.get_condition(y, label2)
        
        # Sampling random timesteps
        t = torch.randint(1, self.timesteps + 1, (batch_size,), device=device)
        
        # Generar ruido
        noise = torch.randn_like(x)
        
        # Aplicar ruido (forward diffusion process)
        x_pert = self.perturb_input(x, t, noise)
        
        # Predecir ruido con el modelo
        pred_noise = self.denoiser(x_pert, t.float() / self.timesteps, c=cond)
        
        # MSE loss entre ruido predicho y real
        loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    def sample(self, y, label2, batch_size=None, data_shape=None):
        """
        Generar muestras mediante reverse diffusion (funciona para imágenes y datos vectorizados)
        Args:
            y: etiquetas multilabel deseadas [batch, y_dim]
            label2: etiquetas categóricas deseadas [batch]
            batch_size: override del batch size 
            data_shape: forma de los datos a generar (auto-detecta si es None)
        Returns:
            generated_data: datos generados (imágenes o vectores)
        """
        if batch_size is None:
            batch_size = y.shape[0]
        
        device = next(self.parameters()).device
        y = y.to(device)
        label2 = label2.to(device)
        
        # Generar condición
        cond = self.get_condition(y, label2)
        
        # Auto-detectar formato basado en el tipo de denoiser
        if data_shape is not None:
            # Usar shape explícita
            x_t = torch.randn(batch_size, *data_shape, device=device)
        else:
            # Auto-detectar basado en el modelo
            if hasattr(self.denoiser, 'input_dim'):
                # MLPUnet: generar datos vectorizados
                x_t = torch.randn(batch_size, self.denoiser.input_dim, device=device)
            elif hasattr(self.denoiser, 'in_channels') and hasattr(self.denoiser, 'h'):
                # ContextUnet: generar imágenes 
                x_t = torch.randn(batch_size, self.denoiser.in_channels, self.denoiser.h, self.denoiser.h, device=device)
            else:
                raise ValueError("No se pudo auto-detectar el formato. Proporciona data_shape explícitamente.")
        
        # Reverse diffusion sampling (DDPM)
        self.denoiser.eval()
        with torch.no_grad():
            for t in range(self.timesteps, 0, -1):
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Predecir ruido
                pred_noise = self.denoiser(x_t, t_tensor.float() / self.timesteps, c=cond)
                
                # Sampling step (simplified DDPM)
                if t > 1:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)
                
                # Reverse step (adaptado a dimensiones dinámicas)
                alpha_t = self.a_t[t]
                alpha_bar_t = self.ab_t[t]
                beta_t = self.b_t[t]
                
                x_t = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise) + beta_t.sqrt() * z
        
        return x_t

class SemisupervisedConditionalDiffusion(ConditionalDiffusion):
    """
    Versión semisupervisada del Conditional Diffusion Model
    Maneja missing values y incluye attribute prediction loss
    """
    def __init__(self, denoiser, attr_predictor, y_dim, y_embed_dim, label2_dim, 
                 timesteps=500, beta1=1e-4, beta2=0.02, alpha=1.0):
        super().__init__(denoiser, y_dim, y_embed_dim, label2_dim, timesteps, beta1, beta2)
        self.attr_predictor = attr_predictor
        self.alpha = alpha  # Weight for attribute prediction loss
    
    def forward(self, x, y, label2):
        """
        Forward pass semisupervisado
        Combina diffusion loss + attribute prediction loss
        """
        # 1. Imputar missing values usando la nueva función común
        y_imputed = impute_missing_labels(x, y, label2, self.attr_predictor, 
                                        self.y_embed, self.label2_dim, 'soft')
        
        # 2. Diffusion loss (usando labels imputados)
        diffusion_loss = super().forward(x, y_imputed, label2)
        
        # 3. Attribute prediction loss (solo para atributos observados)
        attr_loss = compute_attr_prediction_loss(x, y, label2, self.attr_predictor, 
                                                self.y_embed, self.label2_dim)
        
        # 4. Loss combinado
        total_loss = diffusion_loss + self.alpha * attr_loss
        
        return total_loss
    
    def sample(self, y, label2, batch_size=None, data_shape=None, strategy='soft'):
        """
        Generación con manejo de missing values (adaptado para imágenes y datos vectorizados)
        """
        # Imputar missing values usando el attribute predictor
        y_filled = y.clone().float()
        missing_mask = (y == -1)
        
        if missing_mask.any():
            # Auto-detectar dimensión de input para attr_predictor
            if hasattr(self.denoiser, 'input_dim'):
                # MLPUnet: usar input_dim
                input_dim = self.denoiser.input_dim
            elif hasattr(self.denoiser, 'in_channels') and hasattr(self.denoiser, 'h'):
                # ContextUnet: calcular dimensión flatten
                input_dim = self.denoiser.in_channels * self.denoiser.h * self.denoiser.h
            else:
                # Fallback: usar dimensión típica
                input_dim = 9701  # TODO: hacer más dinámico
            
            for i in range(self.y_dim):
                missing_idx = missing_mask[:, i]
                if missing_idx.any():
                    # Crear input dummy para el predictor
                    x_dummy = torch.zeros(missing_idx.sum(), input_dim, device=y.device)
                    y_idx = torch.full((missing_idx.sum(),), i, dtype=torch.long, device=y.device)
                    y_emb = self.y_embed(y_idx)
                    label2_missing = label2[missing_idx]
                    label2_onehot = F.one_hot(label2_missing, num_classes=self.label2_dim).float()
                    attr_input = torch.cat([x_dummy, y_emb, label2_onehot], dim=1)
                    
                    if strategy == 'soft':
                        pred = self.attr_predictor(attr_input)
                        y_filled[missing_idx, i] = pred.squeeze(-1)
                    else:
                        y_filled[missing_idx, i] = 0.5  # neutral
        
        # Usar el método sample del modelo padre con argumentos actualizados
        return super().sample(y_filled, label2, batch_size, data_shape)