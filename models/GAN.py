import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.conditional_utils import get_condition, impute_missing_labels, compute_attr_prediction_loss

############### GAN ###############

# Generation Network (p_theta)
class GenerationNetwork(nn.Module):
    def __init__(self, latent_dim, image_dim, bn = False):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.bn1 = nn.BatchNorm1d(256) if bn else None
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512) if bn else None
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024) if bn else None
        self.fc_out = nn.Linear(1024, image_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.bn1(h) if self.bn1 else h
        h = F.relu(self.fc2(h))
        h = self.bn2(h) if self.bn2 else h
        h = F.relu(self.fc3(h))
        h = self.bn3(h) if self.bn3 else h
        return torch.sigmoid(self.fc_out(h))  # spectra in [0,1]

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


# Discriminator: D(x, y)
class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.fc1 = nn.Linear(image_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        h = F.leaky_relu(self.fc3(h), 0.2)
        return torch.sigmoid(self.fc_out(h))

class ConvDiscriminator(nn.Module):
    def __init__(self, y_embed_dim, label2_dim, img_shape, disc_hidden):
        super().__init__()
        self.img_shape = img_shape
        
        # Parte convolucional para procesar la imagen
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),  # Regularización espacial
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Flatten()
        )
        
        # Calcular dimensión después de convoluciones
        conv_out_dim = (img_shape[1] // 16) * (img_shape[2] // 16) * 256
        
        # Parte fully connected que combina features conv + condición
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim + y_embed_dim + label2_dim, disc_hidden), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(disc_hidden, 256), nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Salida escalar: logits real/fake
        )
    
    def forward(self, x, cond):
        """
        Args:
            x: tensor [batch, D] imágenes flatten (reales o generadas)
            cond: tensor [batch, cond_dim] condición (de get_condition)
        Returns:
            logits: tensor [batch, 1] logits de probabilidad real/fake
        """
        # Reshape a imagen 2D para convoluciones
        x_img = x.view(-1, 1, self.img_shape[1], self.img_shape[2])
        
        # Extraer features convolucionales
        conv_features = self.conv(x_img)
        
        # Combinar con condición
        h = torch.cat([conv_features, cond], dim=1)
        
        return self.fc(h)



################ Conditional GAN ###############

class ConditionalDiscriminator(nn.Module):
    def __init__(self, discriminator_net, y_dim, y_embed_dim, label2_dim):
        super().__init__()
        self.discriminator = discriminator_net
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.label2_dim = label2_dim
    
    def forward(self, x, cond):
        """
        Discriminar si x es real o fake condicionado en cond
        
        Args:
            x: tensor [batch, input_dim] imágenes reales o generadas
            cond: tensor [batch, cond_dim] condición (de get_condition)
        
        Returns:
            logits: tensor [batch, 1] logits de probabilidad real/fake
        """
        logits = self.discriminator(x, cond)
        return logits

class ConditionalGAN(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=128, lambda_g=1.0, lambda_d=1.0):
        """
        GAN Condicional que usa ConditionalDecoder como generador
        
        Args:
            generator: ConditionalDecoder que actúa como generador
            discriminator: ConditionalDiscriminator 
            latent_dim: dimensión del espacio latente para el ruido
            lambda_g: peso para la pérdida del generador
            lambda_d: peso para la pérdida del discriminador
        """
        super().__init__()
        self.generator = generator  # ConditionalDecoder
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.lambda_g = lambda_g
        self.lambda_d = lambda_d
    
    def forward(self, x_real, y, label2):
        """
        Forward pass que calcula ambas pérdidas (discriminador + generador)
        
        Args:
            x_real: tensor [batch, input_dim] datos reales
            y: tensor [batch, num_labels] etiquetas multilabel
            label2: tensor [batch] etiquetas categóricas
        
        Returns:
            total_loss: pérdida total (lambda_d * d_loss + lambda_g * g_loss)
        """
        batch_size = x_real.size(0)
        cond = get_condition(y, label2, self.generator.y_embed, self.generator.label2_dim)
        
        # === PÉRDIDA DEL DISCRIMINADOR ===
        # 1. Discriminador en datos reales
        real_logits = self.discriminator(x_real, cond)
        real_probs = torch.sigmoid(real_logits)
        real_loss = F.binary_cross_entropy(
            real_probs, torch.ones_like(real_probs)
        )
        
        # 2. Generar datos fake
        noise = torch.randn(batch_size, self.latent_dim, device=x_real.device)
        x_fake = self.generator.sample(noise, cond)
        
        # 3. Discriminador en datos fake (para pérdida de D)
        fake_logits_d = self.discriminator(x_fake.detach(), cond)  # detach() para no actualizar G
        fake_probs_d = torch.sigmoid(fake_logits_d)
        fake_loss = F.binary_cross_entropy(
            fake_probs_d, torch.zeros_like(fake_probs_d)
        )
        
        d_loss = real_loss + fake_loss
        
        # === PÉRDIDA DEL GENERADOR ===
        # 4. Discriminador en datos fake (para pérdida de G)
        fake_logits_g = self.discriminator(x_fake, cond)  # sin detach() para actualizar G
        fake_probs_g = torch.sigmoid(fake_logits_g)
        g_loss = F.binary_cross_entropy(
            fake_probs_g, torch.ones_like(fake_probs_g)  # G quiere engañar a D
        )
        
        # === PÉRDIDA TOTAL ===
        total_loss = self.lambda_d * d_loss + self.lambda_g * g_loss
        
        return total_loss
    
    def sample(self, y, label2, batch_size=64):
        """
        Generar muestras condicionadas
        
        Args:
            y: tensor [batch, num_labels] etiquetas multilabel
            label2: tensor [batch] etiquetas categóricas
            batch_size: número de muestras (si y/label2 no especifican batch)
        
        Returns:
            x_fake: tensor [batch, input_dim] muestras generadas
        """
        if y.size(0) != batch_size:
            batch_size = y.size(0)
            
        cond = get_condition(y, label2, self.generator.y_embed, self.generator.label2_dim)
        
        # Generar ruido y decodificar
        noise = torch.randn(batch_size, self.latent_dim, device=y.device)
        x_fake = self.generator.sample(noise, cond)
        
        return x_fake



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