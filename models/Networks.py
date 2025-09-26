import torch
import torch.nn as nn

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
    def __init__(self, latent_dim, img_shape, enc_hidden2, num_layers=3, base_channels=32):
        super().__init__()
        self.img_shape = img_shape
        self.num_layers = num_layers
        self.base_channels = base_channels
        conv_layers = []
        in_channels = 1
        out_channels = base_channels
        length = img_shape[1]
        for i in range(num_layers):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1))
            conv_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels *= 2
            length = (length + 2*1 - 4)//2 + 1  # update length after conv
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
    def __init__(self, latent_dim, img_shape, dec_hidden2, num_layers=3, base_channels=32):
        super().__init__()
        self.img_shape = img_shape
        self.D = img_shape[1]
        self.num_layers = num_layers
        self.base_channels = base_channels
        # Compute the length after all conv layers in encoder
        length = img_shape[1]
        channels = base_channels
        for i in range(num_layers):
            length = (length + 2*1 - 4)//2 + 1
            channels *= 2
        channels = channels // 2
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden2), nn.LeakyReLU(),
            nn.Linear(dec_hidden2, length * channels), nn.LeakyReLU()
        )
        # Build deconv layers
        deconv_layers = []
        in_channels = channels
        out_channels = in_channels // 2
        for i in range(num_layers-1):
            deconv_layers.append(nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0))
            deconv_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels = in_channels // 2
        # Final layer to 1 channel
        deconv_layers.append(nn.ConvTranspose1d(in_channels, 1, 4, stride=2, padding=1, output_padding=1))
        self.deconv = nn.Sequential(*deconv_layers)
    def forward(self, z):
        # Compute length after all conv layers in encoder
        length = self.img_shape[1]
        channels = self.base_channels
        for i in range(self.num_layers):
            length = (length + 2*1 - 4)//2 + 1
            channels *= 2
        channels = channels // 2
        h = self.fc(z)
        h = h.view(-1, channels, length)
        x_rec = self.deconv(h)
        # Ensure output length matches self.img_shape[1] (D)
        # x_rec should be [batch, 1, length]
        if x_rec.dim() == 2:
            # Add channel dimension if missing
            x_rec = x_rec.unsqueeze(1)
        out_len = x_rec.shape[2]
        target_len = self.img_shape[1]
        if out_len > target_len:
            x_rec = x_rec[:, :, :target_len]
        elif out_len < target_len:
            pad_amt = target_len - out_len
            x_rec = torch.nn.functional.pad(x_rec, (0, pad_amt))
        # Remove channel dimension if present
        x_rec = x_rec.view(x_rec.size(0), -1)
        return x_rec  # Garantiza salida [batch, D]
    

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