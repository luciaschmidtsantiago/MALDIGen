import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple MLP encoder/decoder for 1D spectra (no conditioning)
class MLPEncoder1D(nn.Module):
    def __init__(self, D, num_layers, hidden_dim, latent_dim):
        """
        D: input dimension
        num_layers: number of hidden layers (>=2 recommended)
        hidden_dim: base hidden layer size (final bottleneck)
        latent_dim: latent space dimension
        """
        super().__init__()
        layers = []
        in_dim = D
        # Compute multipliers: e.g. for 5 layers: [8, 4, 2, 1]
        multipliers = [2 ** (num_layers - i - 1) for i in range(num_layers)]
        for m in multipliers:
            out_dim = m * hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 2 * latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPDecoder1D(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, D):
        """
        latent_dim: latent space dimension
        num_layers: number of hidden layers (>=2 recommended)
        hidden_dim: base hidden layer size (initial bottleneck)
        D: output dimension
        """
        super().__init__()
        layers = []
        in_dim = latent_dim
        # Compute multipliers: e.g. for 5 layers: [1, 2, 4, 8]
        multipliers = [2 ** i for i in range(num_layers)]
        for m in multipliers:
            out_dim = m * hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, D))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
    
class CNNEncoder1D(nn.Module):
    def __init__(self, latent_dim, img_shape, enc_hidden2, num_layers=5, base_channels=32, max_pool=False):
        super().__init__()
        self.img_shape = img_shape
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.max_pool = max_pool
        conv_layers = []
        in_channels = 1
        out_channels = base_channels
        length = img_shape[1]
        for i in range(num_layers):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1))
            conv_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels *= 2
            length = (length + 2*1 - 4)//2 + 1
            if self.max_pool and (i+1) % 2 == 0 and i != num_layers-1:
                conv_layers.append(nn.MaxPool1d(2, stride=2))
                length = (length - 2)//2 + 1
        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)
        conv_out_dim = length * (in_channels)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, enc_hidden2), nn.LeakyReLU(),
            nn.Linear(enc_hidden2, 2 * latent_dim)
        )
    def forward(self, x):
        x = x.view(-1, 1, self.img_shape[1])
        h = self.conv(x)
        return self.fc(h)
    
class CNNDecoder1D(nn.Module):
    def __init__(self, latent_dim, img_shape, dec_hidden2, num_layers=5, base_channels=32, max_pool=False):
        super().__init__()
        self.img_shape = img_shape
        self.D = img_shape[1]
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.max_pool = max_pool
        # Dynamically compute output shape to match encoder
        length = img_shape[1]
        channels = base_channels
        for i in range(num_layers):
            length = (length + 2*1 - 4)//2 + 1
            channels *= 2
            if max_pool and (i+1) % 2 == 0 and i != num_layers-1:
                length = (length - 2)//2 + 1
        channels = channels // 2
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden2), nn.LeakyReLU(),
            nn.Linear(dec_hidden2, length * channels), nn.LeakyReLU()
        )
        # Build deconv layers (no MaxUnpool1d)
        deconv_layers = []
        in_channels = channels
        out_channels = in_channels // 2
        for i in range(num_layers-1):
            deconv_layers.append(nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0))
            deconv_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels = in_channels // 2
            # Remove MaxUnpool1d, not needed
        # Final layer to 1 channel
        deconv_layers.append(nn.ConvTranspose1d(in_channels, 1, 4, stride=2, padding=1, output_padding=1))
        self.deconv = nn.Sequential(*deconv_layers)
    def forward(self, z):
        # Dynamically compute output shape to match encoder
        length = self.img_shape[1]
        channels = self.base_channels
        for i in range(self.num_layers):
            length = (length + 2*1 - 4)//2 + 1
            channels *= 2
            if self.max_pool and (i+1) % 2 == 0 and i != self.num_layers-1:
                length = (length - 2)//2 + 1
        channels = channels // 2
        h = self.fc(z)
        h = h.view(-1, channels, length)
        x_rec = self.deconv(h)
        # Ensure output length matches self.img_shape[1] (D)
        if x_rec.dim() == 2:
            x_rec = x_rec.unsqueeze(1)
        out_len = x_rec.shape[2]
        target_len = self.img_shape[1]
        if out_len > target_len:
            x_rec = x_rec[:, :, :target_len]
        elif out_len < target_len:
            pad_amt = target_len - out_len
            x_rec = torch.nn.functional.pad(x_rec, (0, pad_amt))
        x_rec = x_rec.view(x_rec.size(0), -1)
        return x_rec  # Garantiza salida [batch, D]
    
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
    def __init__(self, D=6000, latent_dim=128, n_heads=4, n_layers=4, token_dim=128, n_tokens=300):
        super().__init__()
        # Conv stem (local features)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),  # (B,64,3000)
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=10, stride=10),          # (B,128,300)
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, token_dim))

        # Project channels to token_dim
        self.proj = nn.Linear(128, token_dim)

        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model=token_dim, n_heads=n_heads) for _ in range(n_layers)
        ])

        # Latent heads: output 2*latent_dim for VAE
        self.fc_out = nn.Linear(token_dim, 2 * latent_dim)

    def forward(self, x):
        # x: (B, D) → reshape to 1D "image"
        x = x.unsqueeze(1)  # (B,1,6000)
        x = self.conv(x)    # (B,128,300)
        x = x.permute(0, 2, 1)  # (B,300,128)
        x = self.proj(x) + self.pos_emb  # (B,300,token_dim)

        x = self.transformer(x)  # (B,300,token_dim)
        x = x.mean(dim=1)        # pool tokens (B,token_dim)
        # Output [batch, 2*latent_dim] for VAE
        return self.fc_out(x)

# --- CNN + Attention Decoder ---
class CNNAttenDecoder(nn.Module):
    def __init__(self, D=6000, latent_dim=128, n_heads=4, n_layers=4, token_dim=128, n_tokens=300):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        # Map latent → tokens
        self.fc = nn.Linear(latent_dim, n_tokens * token_dim)

        # Transformer stack
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model=token_dim, n_heads=n_heads) for _ in range(n_layers)
        ])

        # Project back to feature maps
        self.to_feat = nn.Linear(token_dim, 128)

        # ConvTranspose pipeline (de-patching)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=10, stride=10),   # (B,64,3000)
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),      # (B,32,6000)
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),  # final output in [0,1]
        )

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc(z).view(-1, self.n_tokens, self.token_dim)  # (B,300,token_dim)
        x = self.transformer(x)  # (B,300,token_dim)
        x = self.to_feat(x)      # (B,300,128)
        x = x.permute(0, 2, 1)   # (B,128,300)
        x = self.deconv(x)       # (B,1,6000)
        return x.squeeze(1)      # (B,6000)
    
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