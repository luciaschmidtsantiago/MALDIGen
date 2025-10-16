import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder1D(nn.Module):
    def __init__(self, input_dim, num_layers, latent_dim, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.in_dim = input_dim + cond_dim

        layers = []

        # Create hidden dims: [8*latent, 4*latent, ..., 4*latent] (last hidden always 4*latent)
        hidden_dims = [2 ** (num_layers + 1 - i) * latent_dim for i in range(num_layers - 1)]
        hidden_dims.append(4 * latent_dim)  # Last hidden layer before projection

        for out_dim in hidden_dims:
            layers.append(nn.Linear(self.in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            self.in_dim = out_dim

        # Final projection to [μ, logσ²]
        layers.append(nn.Linear(self.in_dim, 2 * latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1) # [batch_size, input_dim + cond_dim]
        h = self.net(x) # [batch_size, final_hidden_dim]
        return h

class MLPDecoder1D(nn.Module):
    def __init__(self, latent_dim, num_layers, output_dim, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.in_dim = latent_dim + cond_dim

        layers = []
        for i in range(num_layers):
            out_dim = 2 ** (i + 2) * latent_dim  # e.g., 4×, 8×, 16×...
            layers.append(nn.Linear(self.in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            self.in_dim = out_dim

        layers.append(nn.Linear(self.in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, cond=None):
        if cond is not None:
            z = torch.cat([z, cond], dim=1)  # shape: [batch, latent_dim + cond_dim]
        return self.net(z)
    

class CNNEncoder1D(nn.Module):
    """
    CNN encoder that conditions on a label embedding after convolutional layers when cond_dim > 0.
    """
    def __init__(self, latent_dim, img_shape, num_layers=3, base_channels=32, max_pool=False, cond_dim=0):
        super().__init__()
        self.img_shape = img_shape
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.max_pool = max_pool
        self.cond_dim = cond_dim

        # ----- Convolutional feature extractor -----
        conv_layers = []
        in_channels = 1
        out_channels = base_channels
        length = img_shape[1]

        for i in range(num_layers):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1))
            conv_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels *= 2

            length = (length + 2 * 1 - 4) // 2 + 1  # after stride-2 conv

            if self.max_pool and (i + 1) % 2 == 0 and i != num_layers - 1:
                conv_layers.append(nn.MaxPool1d(2, stride=2))
                length = (length - 2) // 2 + 1

        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)

        # Flattened feature size from CNN
        conv_out_dim = length * in_channels

        # ----- Fully connected projection -----
        # The conditioning vector is concatenated here -> conv_out_dim + cond_dim
        fc_hidden = [4 * latent_dim * (2 ** i) for i in reversed(range(num_layers))]
        fc_layers = []
        in_dim = conv_out_dim + cond_dim

        for out_dim in fc_hidden:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.LeakyReLU())
            in_dim = out_dim

        fc_layers.append(nn.Linear(in_dim, 2 * latent_dim))  # outputs mean + logvar
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, cond=None):
        # x: [batch, D]
        x = x.view(-1, 1, self.img_shape[1])
        h = self.conv(x)  # CNN feature map flattened
        if cond is not None:
            h = torch.cat([h, cond], dim=1)
        return self.fc(h)

class CNNDecoder1D(nn.Module):
    """
    CNN decoder that conditions on label embedding concatenated to the latent vector when cond_dim > 0.
    """
    def __init__(self, latent_dim, img_shape, num_layers=3, base_channels=32, max_pool=False, cond_dim=0):
        super().__init__()
        self.img_shape = img_shape
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.max_pool = max_pool
        self.cond_dim = cond_dim

        # ----- Mirror the encoder’s feature shape -----
        length = img_shape[1]
        channels = base_channels
        pool_count = 0
        for i in range(num_layers):
            length = (length + 2 * 1 - 4) // 2 + 1
            channels *= 2
            if max_pool and (i + 1) % 2 == 0 and i != num_layers - 1:
                length = (length - 2) // 2 + 1
                pool_count += 1
        channels = channels // 2

        # ----- Fully connected expansion -----
        fc_hidden = [4 * latent_dim * (2 ** i) for i in range(num_layers)]
        fc_layers = []
        in_dim = latent_dim + cond_dim  # conditioning added here

        for out_dim in fc_hidden:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.LeakyReLU())
            in_dim = out_dim

        fc_layers.append(nn.Linear(in_dim, length * channels))
        fc_layers.append(nn.LeakyReLU())
        self.fc = nn.Sequential(*fc_layers)

        # ----- Deconvolutional upsampling -----
        num_upsamples = num_layers + pool_count
        deconv_layers = []
        in_channels = channels
        out_channels = in_channels // 2

        for i in range(num_upsamples - 1):
            deconv_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            deconv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            deconv_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels = max(1, in_channels // 2)

        deconv_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        deconv_layers.append(nn.Conv1d(in_channels, 1, kernel_size=3, padding=1))
        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, z, cond=None):
        if cond is not None:
            z = torch.cat([z, cond], dim=1)
        h = self.fc(z)
        # Reshape for Conv1d
        length = self.img_shape[1]
        channels = self.base_channels
        pool_count = 0
        for i in range(self.num_layers):
            length = (length + 2 * 1 - 4) // 2 + 1
            channels *= 2
            if self.max_pool and (i + 1) % 2 == 0 and i != self.num_layers - 1:
                length = (length - 2) // 2 + 1
                pool_count += 1
        channels = channels // 2

        h = h.view(-1, channels, length)
        x_rec = self.deconv(h)

        # Match original dimension
        target_len = self.img_shape[1]
        out_len = x_rec.shape[2]
        if out_len > target_len:
            x_rec = x_rec[:, :, :target_len]
        elif out_len < target_len:
            pad_amt = target_len - out_len
            x_rec = nn.functional.pad(x_rec, (0, pad_amt))

        return x_rec.view(x_rec.size(0), -1)


# --- Transformer blocks for 1D tokens ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop(attn_out)
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x

# --- CNN + Attention Encoder ---
class CNNAttenEncoder(nn.Module):
    def __init__(self, D, latent_dim, n_heads, n_layers, n_tokens=300):
        super().__init__()
        token_dim = 4 * latent_dim  # matches new standard

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=10, stride=10),
            nn.GELU(),
        )

        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, token_dim))
        self.proj = nn.Linear(128, token_dim)

        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model=token_dim, n_heads=n_heads) for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(token_dim, 2 * latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x) + self.pos_emb[:, :x.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc_out(x)

# --- CNN + Attention Decoder ---
class CNNAttenDecoder(nn.Module):
    def __init__(self, D, latent_dim, n_heads, n_layers, n_tokens=300):
        super().__init__()
        self.n_tokens = n_tokens
        token_dim = 4 * latent_dim
        self.token_dim = token_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * latent_dim, token_dim * n_tokens)
        )

        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model=token_dim, n_heads=n_heads) for _ in range(n_layers)
        ])

        self.to_feat = nn.Linear(token_dim, 128)

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=10, stride=10),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, self.n_tokens, self.token_dim)
        x = self.transformer(x)
        x = self.to_feat(x)
        x = x.permute(0, 2, 1)
        x = self.deconv(x)
        return x.squeeze(1)
    
############# CONDITIONING #############

class MLPEncoder_(nn.Module):
    def __init__(self, D, y_embed_dim, label2_dim, enc_hidden1, enc_hidden2, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D + y_embed_dim + label2_dim, enc_hidden1), nn.LeakyReLU(),
            nn.Linear(enc_hidden1, enc_hidden2), nn.LeakyReLU(),
            nn.Linear(enc_hidden2, 2 * latent_dim)
        )
    def forward(self, x, cond):
        h = torch.cat([x, cond], dim=1)
        return self.fc(h)

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, y_embed_dim, label2_dim, dec_hidden1, dec_hidden2, D):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + y_embed_dim + label2_dim, dec_hidden2), nn.LeakyReLU(),
            nn.Linear(dec_hidden2, dec_hidden1), nn.LeakyReLU(),
            nn.Linear(dec_hidden1, D)
        )
    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=1)
        return self.fc(h)


class ConvEncoder(nn.Module):
    def __init__(self, y_embed_dim, label2_dim, latent_dim, img_shape, enc_hidden2):
        super().__init__()
        self.img_shape = img_shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Flatten()
        )
        conv_out_dim = (img_shape[1] // 8) * (img_shape[2] // 8) * 128
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim + y_embed_dim + label2_dim, enc_hidden2), nn.LeakyReLU(),
            nn.Linear(enc_hidden2, 2 * latent_dim)
        )
    def forward(self, x, cond):
        x = x.view(-1, 1, self.img_shape[1], self.img_shape[2])
        h = self.conv(x)
        h = torch.cat([h, cond], dim=1)
        return self.fc(h)

class ConvDecoder(nn.Module):
    def __init__(self, y_embed_dim, label2_dim, latent_dim, img_shape, dec_hidden2):
        super().__init__()
        self.img_shape = img_shape
        self.D = img_shape[1] * img_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + y_embed_dim + label2_dim, dec_hidden2), nn.LeakyReLU(),
            nn.Linear(dec_hidden2, (img_shape[1] // 8) * (img_shape[2] // 8) * 128), nn.LeakyReLU()
        )
        # Ajusta output_padding en la primera y última capa para obtener [batch, 1, 109, 89]
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, output_padding=1)
        )
    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=1)
        h = self.fc(h)
        h = h.view(-1, 128, self.img_shape[1] // 8, self.img_shape[2] // 8)
        x_rec = self.deconv(h)
        # Recorta si sobra (resulta en [batch, 1, 109, 93], así que se quitan 4 columnas)
        x_rec = x_rec[:, :, :self.img_shape[1], :self.img_shape[2]]
        return x_rec.reshape(-1, self.D)  # Garantiza salida [batch, D]